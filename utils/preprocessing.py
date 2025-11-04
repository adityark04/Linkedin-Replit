"""
Data Preprocessing Utilities
Handles tokenization, vocabulary building, and dataset preparation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import re
from collections import Counter


class Tokenizer:
    """Simple word-level tokenizer for LinkedIn posts."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_built = False
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and clean
        text = text.lower()
        text = re.sub(r'[^\w\s#@]', ' ', text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        # Count word frequencies
        word_counts = Counter(all_tokens)
        
        # Add most common words to vocabulary
        most_common = word_counts.most_common(self.vocab_size - len(self.word2idx))
        
        for word, count in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"✓ Vocabulary built with {len(self.word2idx)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.word2idx[self.sos_token]] + ids + [self.word2idx[self.eos_token]]
        
        # Truncate if too long
        if len(ids) > self.max_length:
            ids = ids[:self.max_length-1] + [self.word2idx[self.eos_token]]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for idx in ids:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in [self.pad_token, self.unk_token, self.sos_token, self.eos_token]:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def pad_sequence(self, ids: List[int], max_length: int = None) -> List[int]:
        """Pad sequence to max_length."""
        if max_length is None:
            max_length = self.max_length
        
        if len(ids) < max_length:
            ids = ids + [self.word2idx[self.pad_token]] * (max_length - len(ids))
        
        return ids[:max_length]


class PostDataset(Dataset):
    """Dataset for LinkedIn post optimization."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        source_column: str = 'content',
        target_column: str = 'content',
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with posts
            tokenizer: Tokenizer instance
            source_column: Column name for source text (drafts)
            target_column: Column name for target text (optimized)
            max_length: Maximum sequence length
        """
        self.df = df
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.target_column = target_column
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get source and target texts
        source_text = str(row[self.source_column])
        target_text = str(row[self.target_column])
        
        # Encode
        source_ids = self.tokenizer.encode(source_text)
        target_ids = self.tokenizer.encode(target_text)
        
        # Pad
        source_ids = self.tokenizer.pad_sequence(source_ids, self.max_length)
        target_ids = self.tokenizer.pad_sequence(target_ids, self.max_length)
        
        return {
            'source': torch.tensor(source_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text
        }


def create_dataloaders(
    df: pd.DataFrame,
    tokenizer: Tokenizer,
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        df: DataFrame with posts
        tokenizer: Tokenizer instance
        batch_size: Batch size
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Shuffle if requested
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Create datasets
    train_dataset = PostDataset(train_df, tokenizer)
    val_dataset = PostDataset(val_df, tokenizer)
    test_dataset = PostDataset(test_df, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    sources = torch.stack([item['source'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return sources, targets
