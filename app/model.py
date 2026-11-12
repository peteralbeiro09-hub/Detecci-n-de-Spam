"""Wrapper around the pre-trained Hugging Face model Goodmotion/spam-mail-classifier."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "Goodmotion/spam-mail-classifier"
MAX_LENGTH = 128
# Index order used by the model author: 0 = NOSPAM, 1 = SPAM.
LABELS = ["ham", "spam"]


class SpamClassifier:
    """Loads the tokenizer/model once and returns SPAM probabilities for a list of texts."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Return an array with the SPAM probability of every text."""
        texts = [str(t) if t is not None else "" for t in texts]
        probs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def predict(self, texts: Iterable[str], threshold: float = 0.5, batch_size: int = 32):
        """Return (labels, spam_probabilities) using the given decision threshold."""
        p = self.predict_proba(texts, batch_size=batch_size)
        labels = np.where(p >= threshold, LABELS[1], LABELS[0])
        return labels, p
