"""
BrowseComp Dataset Module

Handles downloading, decrypting, and preparing the BrowseComp dataset.
Based on OpenAI's browse_comp_test_set.csv format.
"""

import base64
import hashlib
import pandas as pd
import dspy
from typing import List, Optional
import random


class BrowseCompDataset:
    """BrowseComp dataset handler with decryption support."""
    
    DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    def __init__(self, num_examples: Optional[int] = None, seed: int = 42):
        """
        Initialize BrowseComp dataset.
        
        Args:
            num_examples: Limit number of examples (None for all)
            seed: Random seed for sampling
        """
        self.num_examples = num_examples
        self.seed = seed
        self._examples = None
        
    def _derive_key(self, password: str, length: int) -> bytes:
        """Derive a fixed-length key from the password using SHA256."""
        hasher = hashlib.sha256()
        hasher.update(password.encode())
        key = hasher.digest()
        return key * (length // len(key)) + key[: length % len(key)]
    
    def _decrypt(self, ciphertext_b64: str, password: str) -> str:
        """Decrypt base64-encoded ciphertext with XOR."""
        encrypted = base64.b64decode(ciphertext_b64)
        key = self._derive_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        return decrypted.decode()
    
    def load(self) -> List[dspy.Example]:
        """Load and decrypt the dataset."""
        if self._examples is not None:
            return self._examples
            
        # Download dataset
        df = pd.read_csv(self.DATASET_URL)
        
        # Process each row
        examples = []
        for _, row in df.iterrows():
            # Decrypt problem and answer using canary
            problem = self._decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = self._decrypt(row.get("answer", ""), row.get("canary", ""))
            
            # Create DSPy Example
            example = dspy.Example(
                problem=problem,
                answer=answer,
                canary=row.get("canary", "")  # Keep for reference
            ).with_inputs("problem")
            
            examples.append(example)
        
        # Sample if requested
        if self.num_examples:
            random.seed(self.seed)
            examples = random.sample(examples, min(self.num_examples, len(examples)))
            
        self._examples = examples
        return examples
    
    def __len__(self):
        """Return number of examples."""
        if self._examples is None:
            self.load()
        return len(self._examples)
    
    def __getitem__(self, idx):
        """Get example by index."""
        if self._examples is None:
            self.load()
        return self._examples[idx]
    
    def split(self, train_size: float = 0.8):
        """
        Split dataset into train/test sets.
        
        Args:
            train_size: Fraction for training set
            
        Returns:
            train_examples, test_examples
        """
        examples = self.load()
        random.seed(self.seed)
        shuffled = random.sample(examples, len(examples))
        
        split_idx = int(len(shuffled) * train_size)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]
        
        return train, test
