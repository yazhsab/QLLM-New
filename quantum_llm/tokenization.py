"""
Tokenization utilities for Quantum Large Language Models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
import os
import json
from collections import Counter


class QLLMTokenizer:
    """
    Tokenizer for Quantum Large Language Models.
    """
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 pad_token: str = "[PAD]",
                 unk_token: str = "[UNK]",
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]",
                 max_length: int = 128):
        """
        Initialize the QLLM tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_length = max_length
        
        # Initialize vocabulary
        self.token_to_id = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Special token IDs
        self.pad_token_id = self.token_to_id[pad_token]
        self.unk_token_id = self.token_to_id[unk_token]
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
    
    def train_from_texts(self, texts: List[str], min_freq: int = 2):
        """
        Train the tokenizer from a list of texts.
        
        Args:
            texts: List of text documents
            min_freq: Minimum frequency for a token to be included
        """
        # Simple word-level tokenization
        all_words = []
        for text in texts:
            # Split by whitespace and punctuation
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter by minimum frequency
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= min_freq}
        
        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top vocab_size - len(special_tokens)
        n_special = len(self.token_to_id)
        vocab_words = [word for word, _ in sorted_words[:self.vocab_size - n_special]]
        
        # Add to vocabulary
        for i, word in enumerate(vocab_words):
            self.token_to_id[word] = i + n_special
            self.id_to_token[i + n_special] = word
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word-level tokenization
        return re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    def encode(self, 
              text: str, 
              add_special_tokens: bool = True,
              padding: bool = True,
              truncation: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            List of token IDs
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Truncate
        if truncation and len(tokens) > self.max_length:
            if add_special_tokens:
                tokens = tokens[:self.max_length-1] + [self.eos_token]
            else:
                tokens = tokens[:self.max_length]
        
        # Convert to IDs
        ids = self.convert_tokens_to_ids(tokens)
        
        # Pad
        if padding and len(ids) < self.max_length:
            ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
        
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
        # Convert to tokens
        tokens = self.convert_ids_to_tokens(ids)
        
        # Skip special tokens
        if skip_special_tokens:
            special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
            tokens = [token for token in tokens if token not in special_tokens]
        
        # Join tokens
        text = ' '.join(tokens)
        
        # Fix spacing for punctuation
        text = re.sub(r'\s([.,!?;:])', r'\1', text)
        
        return text
    
    def batch_encode(self, 
                    texts: List[str], 
                    add_special_tokens: bool = True,
                    padding: bool = True,
                    truncation: bool = True) -> torch.Tensor:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            Tensor of token IDs [batch_size, seq_len]
        """
        batch_ids = [self.encode(text, add_special_tokens, padding, truncation) 
                    for text in texts]
        return torch.tensor(batch_ids, dtype=torch.long)
    
    def save(self, filepath: str):
        """
        Save the tokenizer to a file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data to save
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'max_length': self.max_length,
            'token_to_id': self.token_to_id
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a tokenizer from a file.
        
        Args:
            filepath: Path to load the tokenizer from
            
        Returns:
            QLLMTokenizer: Loaded tokenizer
        """
        # Load from file
        with open(filepath, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Create new instance
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            pad_token=tokenizer_data['pad_token'],
            unk_token=tokenizer_data['unk_token'],
            bos_token=tokenizer_data['bos_token'],
            eos_token=tokenizer_data['eos_token'],
            max_length=tokenizer_data['max_length']
        )
        
        # Load vocabulary
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = {int(id): token for token, id in tokenizer.token_to_id.items()}
        
        return tokenizer


class QLLMDataset(torch.utils.data.Dataset):
    """
    Dataset for Quantum Large Language Models.
    """
    
    def __init__(self, 
                texts: List[str], 
                tokenizer: QLLMTokenizer,
                block_size: int = 128):
        """
        Initialize the QLLM dataset.
        
        Args:
            texts: List of text documents
            tokenizer: Tokenizer to use
            block_size: Size of text blocks
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize all texts
        self.examples = []
        
        for text in texts:
            # Encode text
            input_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,
                padding=False,
                truncation=False
            )
            
            # Create blocks
            for i in range(0, len(input_ids) - block_size, block_size // 2):
                block = input_ids[i:i + block_size]
                if len(block) == block_size:
                    self.examples.append(block)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Get input_ids
        input_ids = self.examples[idx][:-1]
        
        # Get labels (shifted right)
        labels = self.examples[idx][1:]
        
        return torch.tensor(input_ids), torch.tensor(labels)