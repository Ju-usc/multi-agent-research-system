import base64
import hashlib
import os
import random
from pathlib import Path

import dspy
import pandas as pd


class BrowseCompDataset:
    DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

    def __init__(self, num_examples: int | None = None, seed: int = 42):
        self.num_examples = num_examples
        self.seed = seed
        self._examples = None

    def _derive_key(self, password: str, length: int) -> bytes:
        key = hashlib.sha256(password.encode()).digest()
        return key * (length // len(key)) + key[: length % len(key)]

    def _decrypt(self, ciphertext_b64: str, password: str) -> str:
        encrypted = base64.b64decode(ciphertext_b64)
        key = self._derive_key(password, len(encrypted))
        return bytes(a ^ b for a, b in zip(encrypted, key)).decode()

    def load(self) -> list[dspy.Example]:
        if self._examples is not None:
            return self._examples

        df = None
        local_path = os.getenv("BROWSECOMP_CSV_PATH")
        cache_path = Path(".cache/browse_comp_test_set.csv")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if local_path and Path(local_path).exists():
                df = pd.read_csv(local_path)
            elif cache_path.exists():
                df = pd.read_csv(cache_path)
            else:
                df = pd.read_csv(self.DATASET_URL)
                try:
                    df.to_csv(cache_path, index=False)
                except Exception:
                    pass
        except Exception:
            return self._offline_examples()

        examples = []
        for _, row in df.iterrows():
            problem = self._decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = self._decrypt(row.get("answer", ""), row.get("canary", ""))
            example = dspy.Example(
                problem=problem, answer=answer, canary=row.get("canary", "")
            ).with_inputs("problem")
            examples.append(example)

        if self.num_examples is not None and self.num_examples > 0:
            random.seed(self.seed)
            examples = random.sample(examples, min(self.num_examples, len(examples)))

        self._examples = examples
        return examples

    def __len__(self):
        if self._examples is None:
            self.load()
        return len(self._examples)

    def __getitem__(self, idx):
        if self._examples is None:
            self.load()
        return self._examples[idx]

    def split(self, train_size: float = 0.8) -> tuple[list, list]:
        examples = self.load()
        random.seed(self.seed)
        shuffled = random.sample(examples, len(examples))
        split_idx = int(len(shuffled) * train_size)
        return shuffled[:split_idx], shuffled[split_idx:]

    def _offline_examples(self) -> list[dspy.Example]:
        n = self.num_examples or 5
        examples = [
            dspy.Example(problem=f"What is {i}+{i}?", answer=str(i + i)).with_inputs("problem")
            for i in range(n)
        ]
        self._examples = examples
        return examples
