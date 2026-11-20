"""A small, local n-gram language model with add-k smoothing and backoff."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
SENTENCE_END = {".", "!", "?"}


def preprocess(text: str) -> list[list[str]]:
    """Normalize text into sentence token lists, retaining useful punctuation."""
    text = text.replace("’", "'").replace("\r", "\n")
    sentences: list[list[str]] = []
    current: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if token.isspace():
            continue
        current.append(token)
        if token in SENTENCE_END:
            if current:
                sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [sentence for sentence in sentences if sentence]


def load_corpus(path: str | Path) -> list[str]:
    """Read UTF-8 .txt files from a file or directory, in stable path order."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {source}")
    files = [source] if source.is_file() else sorted(source.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in corpus path: {source}")
    texts = []
    for file in files:
        if file.suffix.lower() == ".txt":
            texts.append(file.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"No .txt files found in corpus path: {source}")
    return texts


class NGramModel:
    """Count-based language model. `order=3` is a trigram model."""

    def __init__(self, order: int = 3, k: float = 0.1, seed: int | None = None):
        if order < 1:
            raise ValueError("order must be at least 1")
        if k <= 0:
            raise ValueError("add-k smoothing value k must be greater than 0")
        self.order = order
        self.k = float(k)
        self.random = random.Random(seed)
        self.vocabulary: set[str] = set()
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._trained = False

    def fit(self, texts: Iterable[str]) -> "NGramModel":
        self.vocabulary.clear()
        self.counts.clear()
        for text in texts:
            for sentence in preprocess(text):
                self.vocabulary.update(sentence)
                padded = ["<s>"] * (self.order - 1) + sentence + ["</s>"]
                for i in range(self.order - 1, len(padded)):
                    for width in range(self.order):
                        context = tuple(padded[max(0, i - width):i]) if width else ()
                        self.counts[context][padded[i]] += 1
        if not self.vocabulary:
            raise ValueError("Training text contains no tokens")
        self._trained = True
        return self

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary) + 1  # include the end-of-sentence token

    def _context_counts(self, context: Sequence[str]) -> Counter[str]:
        for width in range(min(self.order - 1, len(context)), -1, -1):
            key = tuple(context[-width:]) if width else ()
            if key in self.counts:
                return self.counts[key]
        return Counter()

    def probabilities(self, context: Sequence[str]) -> dict[str, float]:
        """Return add-k probabilities for candidate next tokens, with backoff."""
        if not self._trained:
            raise RuntimeError("Call fit() before requesting predictions")
        normalized = [token.lower() for token in context]
        counts = self._context_counts(normalized)
        candidates = self.vocabulary | {"</s>"}
        denominator = sum(counts.values()) + self.k * self.vocabulary_size
        return {word: (counts[word] + self.k) / denominator for word in candidates}

    def predict_next(self, context: Sequence[str]) -> str:
        """Sample a next token from the smoothed distribution."""
        probs = self.probabilities(context)
        words = list(probs)
        return self.random.choices(words, weights=[probs[word] for word in words], k=1)[0]

    def generate(self, prompt: str, max_words: int = 30) -> str:
        """Append up to max_words sampled words, stopping at sentence end."""
        if max_words < 1:
            raise ValueError("max_words must be at least 1")
        if not self._trained:
            raise RuntimeError("Call fit() before generating text")
        prompt_tokens = TOKEN_RE.findall(prompt.lower())
        if not prompt_tokens:
            raise ValueError("Please enter a prompt containing at least one word")
        generated: list[str] = []
        context = prompt_tokens[:]
        for _ in range(max_words):
            probs = self.probabilities(context)
            normalized_context = [part.lower() for part in context]
            observed = self._observed_context_counts(normalized_context)
            # Keep sampled continuations anchored to patterns in the corpus when
            # such patterns exist. Smoothing still determines their probabilities;
            # the full smoothed distribution remains available via probabilities().
            candidates = [word for word in probs if observed[word] > 0] or list(probs)
            token = self.random.choices(
                candidates, weights=[probs[word] for word in candidates], k=1
            )[0]
            if token == "</s>":
                break
            generated.append(token)
            context.append(token)
            if token in SENTENCE_END:
                break
        continuation = _detokenize(generated)
        separator = "" if generated and generated[0] in {".", ",", "!", "?", ";", ":", ")", "]"} else " "
        return prompt.rstrip() + (separator + continuation if continuation else "")

    def _observed_context_counts(self, context: Sequence[str]) -> Counter[str]:
        """Find the longest observed suffix; if absent, a known prompt word."""
        max_width = min(self.order - 1, len(context))
        for width in range(max_width, 0, -1):
            counts = self.counts.get(tuple(context[-width:]))
            if counts:
                return counts
        for token in reversed(context):
            counts = self.counts.get((token,))
            if counts:
                return counts
        return self.counts.get((), Counter())


def _detokenize(tokens: Sequence[str]) -> str:
    result = ""
    for token in tokens:
        if token in {".", ",", "!", "?", ";", ":", ")", "]"}:
            result = result.rstrip() + token
        elif token in {"(", "["}:
            result += (" " if result else "") + token
        else:
            result += (" " if result else "") + token
    return result
