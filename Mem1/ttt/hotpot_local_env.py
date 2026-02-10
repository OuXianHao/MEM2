import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class EvidenceItem:
    title: str
    text: str
    score: float


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class HotpotQALocalEnv:
    """Local retrieval environment for HotpotQA distractor context only."""

    def __init__(self, top_k_evidence: int = 3, max_evidence_chars: int = 1200):
        self.top_k_evidence = top_k_evidence
        self.max_evidence_chars = max_evidence_chars

    def _flatten_context(self, context: Sequence[Sequence]) -> List[Dict[str, str]]:
        docs = []
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            title, sentences = item
            if not isinstance(sentences, list):
                continue
            text = " ".join([s.strip() for s in sentences if isinstance(s, str) and s.strip()]).strip()
            if not text:
                continue
            docs.append({"title": str(title).strip(), "text": text})
        return docs

    def _score(self, query: str, doc_text: str) -> float:
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(doc_text)
        if not q_tokens or not d_tokens:
            return 0.0
        q_counter = Counter(q_tokens)
        d_counter = Counter(d_tokens)
        overlap = sum(min(q_counter[t], d_counter[t]) for t in q_counter)
        norm = (len(set(q_tokens)) * len(set(d_tokens))) ** 0.5
        return overlap / max(norm, 1e-6)

    def search(self, context: Sequence[Sequence], query: str) -> str:
        docs = self._flatten_context(context)
        scored: List[EvidenceItem] = []
        for d in docs:
            score = self._score(query, d["title"] + " " + d["text"])
            scored.append(EvidenceItem(title=d["title"], text=d["text"], score=score))

        scored.sort(key=lambda x: x.score, reverse=True)

        snippets = []
        seen = set()
        char_count = 0
        for ev in scored[: self.top_k_evidence * 2]:
            key = (ev.title, ev.text[:160])
            if key in seen:
                continue
            seen.add(key)
            chunk = f"Title: {ev.title}\n{ev.text}"
            if char_count + len(chunk) > self.max_evidence_chars:
                remain = self.max_evidence_chars - char_count
                if remain <= 0:
                    break
                chunk = chunk[:remain]
            snippets.append(chunk)
            char_count += len(chunk) + 2
            if len(snippets) >= self.top_k_evidence or char_count >= self.max_evidence_chars:
                break

        return "<information>\n" + "\n\n".join(snippets) + "\n</information>"
