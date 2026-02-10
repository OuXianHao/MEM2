import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


@dataclass
class UpdateResult:
    updated: bool
    uncertainty: Optional[float]
    loss: Optional[float]
    teacher_action: str
    time_update: float


class StepTTTUpdater:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        optimizer,
        max_grad_norm: float,
        steps_per_update: int,
        gating_low: float,
        gating_high: float,
        stop_token_ids: Sequence[int],
        max_action_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.steps_per_update = steps_per_update
        self.gating_low = gating_low
        self.gating_high = gating_high
        self.stop_token_ids = list(stop_token_ids)
        self.max_action_tokens = max_action_tokens

    def _build_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.no_grad()
    def _teacher_generate(self, prompt: str):
        self.model.eval()
        inputs = self._build_input(prompt)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_action_tokens,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        full_seq = output.sequences[0]
        new_tokens = full_seq[prompt_len:]
        action_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        transition_scores = self.model.compute_transition_scores(
            output.sequences,
            output.scores,
            normalize_logits=True,
        )[0]
        gen_len = max(int(new_tokens.shape[0]), 1)
        uncertainty = float((-transition_scores[:gen_len].mean()).item())
        return action_text, new_tokens, uncertainty

    def _sft_loss_for_action(self, prompt: str, action_token_ids: torch.Tensor) -> torch.Tensor:
        in_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        in_ids = in_ids.to(self.device)
        action_token_ids = action_token_ids.unsqueeze(0).to(self.device)
        full = torch.cat([in_ids, action_token_ids], dim=1)

        labels = torch.full_like(full, -100)
        labels[:, in_ids.shape[1]:] = full[:, in_ids.shape[1]:]

        out = self.model(input_ids=full, labels=labels)
        return out.loss

    def maybe_update(self, prompt: str) -> UpdateResult:
        t0 = time.time()
        teacher_action, teacher_action_ids, uncertainty = self._teacher_generate(prompt)

        should_update = self.gating_low <= uncertainty <= self.gating_high
        if not should_update:
            return UpdateResult(
                updated=False,
                uncertainty=uncertainty,
                loss=None,
                teacher_action=teacher_action,
                time_update=0.0,
            )

        self.model.train()
        loss_value = None
        for _ in range(self.steps_per_update):
            loss = self._sft_loss_for_action(prompt, teacher_action_ids)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            loss_value = float(loss.item())
        self.model.eval()

        return UpdateResult(
            updated=True,
            uncertainty=uncertainty,
            loss=loss_value,
            teacher_action=teacher_action,
            time_update=time.time() - t0,
        )
