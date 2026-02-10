#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Mem1.ttt import HotpotQALocalEnv, StepTTTUpdater


ACTION_PATTERN = {
    "search": re.compile(r"<search>(.*?)</search>", re.DOTALL),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
}



def import_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "peft is required for this script. Please install with `pip install peft`."
        ) from exc
    return LoraConfig, TaskType, get_peft_model

PROMPT_TEMPLATE = """You will answer a complex question through iterative reasoning and local evidence retrieval.

At each step, output in the format:
<think><summary>...</summary><reasoning>...</reasoning></think><search>...</search>
or
<think><summary>...</summary><reasoning>...</reasoning></think><answer>...</answer>

Use <search> to request more information. The environment will return <information>...</information> from local context only.
When you are confident, output <answer> with a concise final answer.

Question: {question}
"""


@dataclass
class StepRecord:
    episode_id: str
    step_id: int
    action_type: str
    action_text: str
    search_query: Optional[str]
    information_snippet: Optional[str]
    teacher_uncertainty: Optional[float]
    updated: bool
    loss: Optional[float]
    time_gen: float
    time_update: float
    forced_terminate: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mode", choices=["baseline", "ttt"], required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_steps_per_episode", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--top_k_evidence", type=int, default=3)
    parser.add_argument("--max_evidence_chars", type=int, default=1200)
    parser.add_argument("--steps_per_update", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gating_low", type=float, default=0.5)
    parser.add_argument("--gating_high", type=float, default=3.0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hotpot_no_gold(path: str, max_samples: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    processed = []
    for row in data[:max_samples]:
        processed.append(
            {
                "episode_id": row.get("_id", str(len(processed))),
                "question": row.get("question", ""),
                "context": row.get("context", []),
            }
        )
    return processed


def parse_action(text: str):
    a_match = ACTION_PATTERN["answer"].search(text)
    if a_match:
        ans = a_match.group(1).strip()
        return "answer", ans

    s_match = ACTION_PATTERN["search"].search(text)
    if s_match:
        q = s_match.group(1).strip()
        return "search", q

    return "invalid", text.strip()


def build_state_prompt(question: str, history: List[str]) -> str:
    return PROMPT_TEMPLATE.format(question=question) + "\n" + "\n".join(history)


def generate_action(model, tokenizer, prompt: str, max_tokens: int, temperature: float, stop_token_ids: List[int], device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature <= 0:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.95))

    t0 = time.time()
    out = model.generate(**gen_kwargs)
    elapsed = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return text, elapsed


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LoraConfig, TaskType, get_peft_model = import_peft()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    stop_texts = ["</search>", "</answer>"]
    stop_token_ids = []
    for t in stop_texts:
        tok = tokenizer.encode(t, add_special_tokens=False)
        if len(tok) == 1:
            stop_token_ids.append(tok[0])
    if not stop_token_ids and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    updater = StepTTTUpdater(
        model=model,
        tokenizer=tokenizer,
        device=device,
        optimizer=optimizer,
        max_grad_norm=args.max_grad_norm,
        steps_per_update=args.steps_per_update,
        gating_low=args.gating_low,
        gating_high=args.gating_high,
        stop_token_ids=stop_token_ids,
        max_action_tokens=args.max_tokens,
    )

    env = HotpotQALocalEnv(top_k_evidence=args.top_k_evidence, max_evidence_chars=args.max_evidence_chars)
    episodes = load_hotpot_no_gold(args.data_path, args.max_samples)

    trace_path = os.path.join(args.out_dir, "episode_trace.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    total_steps = 0
    total_updates = 0
    uncertainty_values = []
    loss_values = []
    search_steps = 0
    forced_terminated = 0
    t_all = time.time()

    with open(trace_path, "w", encoding="utf-8") as wf:
        for ep in episodes:
            question = ep["question"]
            context = ep["context"]
            history: List[str] = []
            ended = False
            forced = False

            for step_id in range(args.max_steps_per_episode):
                state_prompt = build_state_prompt(question, history)

                ttt_uncertainty = None
                ttt_updated = False
                ttt_loss = None
                t_update = 0.0
                if args.mode == "ttt" and history:
                    update_result = updater.maybe_update(state_prompt)
                    ttt_uncertainty = update_result.uncertainty
                    ttt_updated = update_result.updated
                    ttt_loss = update_result.loss
                    t_update = update_result.time_update
                    if ttt_uncertainty is not None:
                        uncertainty_values.append(ttt_uncertainty)
                    if ttt_updated:
                        total_updates += 1
                    if ttt_loss is not None:
                        loss_values.append(ttt_loss)

                model.eval()
                action_raw, t_gen = generate_action(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=state_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop_token_ids=stop_token_ids,
                    device=device,
                )
                action_type, action_text = parse_action(action_raw)
                info = None
                search_query = None

                if action_type == "search":
                    search_query = action_text
                    info = env.search(context=context, query=search_query)
                    history.append(action_raw)
                    history.append(info)
                    search_steps += 1
                elif action_type == "answer":
                    history.append(action_raw)
                    ended = True
                else:
                    info = env.search(context=context, query=question)
                    fallback = "<search>" + question + "</search>"
                    action_type = "search"
                    action_text = question
                    search_query = question
                    history.append(fallback)
                    history.append(info)
                    search_steps += 1

                if step_id == args.max_steps_per_episode - 1 and not ended:
                    forced = True
                    forced_terminated += 1
                    forced_answer = "<answer>unknown</answer>"
                    history.append(forced_answer)
                    action_type = "answer"
                    action_text = "unknown"
                    ended = True

                rec = StepRecord(
                    episode_id=ep["episode_id"],
                    step_id=step_id,
                    action_type=action_type,
                    action_text=action_text,
                    search_query=search_query,
                    information_snippet=info[:300] if info else None,
                    teacher_uncertainty=ttt_uncertainty,
                    updated=ttt_updated,
                    loss=ttt_loss,
                    time_gen=t_gen,
                    time_update=t_update,
                    forced_terminate=forced,
                )
                wf.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

                total_steps += 1
                if ended:
                    break

    summary = {
        "total_episodes": len(episodes),
        "total_steps": total_steps,
        "num_updates": total_updates,
        "update_rate": (total_updates / max(total_steps, 1)),
        "avg_uncertainty": (sum(uncertainty_values) / len(uncertainty_values)) if uncertainty_values else None,
        "avg_loss": (sum(loss_values) / len(loss_values)) if loss_values else None,
        "avg_search_steps_per_episode": search_steps / max(len(episodes), 1),
        "forced_terminate_rate": forced_terminated / max(len(episodes), 1),
        "total_time": time.time() - t_all,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
