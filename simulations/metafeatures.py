from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from simulations.io_utils import iter_jsonl

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _estimate_zipf_alpha_from_counts(token_counts: Counter[str]) -> float:
    freqs = sorted((freq for freq in token_counts.values() if freq > 0), reverse=True)
    if len(freqs) < 2:
        return 1.0
    xs = [math.log(rank + 1.0) for rank in range(len(freqs))]
    ys = [math.log(float(freq)) for freq in freqs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        return 1.0
    slope = numerator / denominator
    return max(0.01, -slope)


def extract_metafeatures_from_texts(
    texts: Iterable[str],
    max_tokens: int = 2_000_000,
) -> Dict[str, float]:
    token_counts: Counter[str] = Counter()
    seq_lengths: List[int] = []
    total_tokens = 0
    total_chars = 0

    for text in texts:
        if not text:
            continue
        tokens = tokenize(text)
        if not tokens:
            continue
        remaining_budget = max_tokens - total_tokens
        if remaining_budget <= 0:
            break
        if len(tokens) > remaining_budget:
            tokens = tokens[:remaining_budget]
            text = " ".join(tokens)
        token_counts.update(tokens)
        seq_lengths.append(len(tokens))
        total_tokens += len(tokens)
        total_chars += len(text)
        if total_tokens >= max_tokens:
            break

    if total_tokens == 0:
        return {
            "ttr_score": 0.0,
            "hapax_ratio": 0.0,
            "zipf_alpha": 1.0,
            "mean_seq_len": 0.0,
            "std_seq_len": 0.0,
            "token_entropy": 0.0,
            "char_per_token": 0.0,
            "tokens_sampled": 0.0,
            "examples_sampled": 0.0,
        }

    unique_tokens = len(token_counts)
    hapax = sum(1 for _, freq in token_counts.items() if freq == 1)
    probs = [freq / total_tokens for freq in token_counts.values()]
    token_entropy = -sum(prob * math.log(prob, 2) for prob in probs if prob > 0.0)

    return {
        "ttr_score": unique_tokens / total_tokens,
        "hapax_ratio": hapax / max(unique_tokens, 1),
        "zipf_alpha": _estimate_zipf_alpha_from_counts(token_counts),
        "mean_seq_len": mean(seq_lengths),
        "std_seq_len": pstdev(seq_lengths) if len(seq_lengths) > 1 else 0.0,
        "token_entropy": token_entropy,
        "char_per_token": total_chars / max(total_tokens, 1),
        "tokens_sampled": float(total_tokens),
        "examples_sampled": float(len(seq_lengths)),
    }


def extract_metafeatures_from_jsonl(path: Path | str, max_tokens: int = 2_000_000) -> Dict[str, float]:
    def _texts() -> Iterable[str]:
        for row in iter_jsonl(path):
            text = row.get("text")
            if isinstance(text, str):
                yield text

    return extract_metafeatures_from_texts(_texts(), max_tokens=max_tokens)

