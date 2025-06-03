"""
Modality-specific encoders for Quantum LLM.

This module provides encoders for different input modalities (text, images, audio)
that convert raw inputs into embeddings suitable for quantum processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class ModalityEncoder(nn.Module):
    """Base class for all modality encoders."""
    
    def __init__(self, output_dim: int):
        """
        Initialize the modality encoder.
        
        Args:
            output_dim: Dimension of the output embedding
        """
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data
            
        Returns:
            torch.Tensor: Encoded representation
        """
        raise NotImplementedError("Subclasses must implement forward method")


class TextEncoder(ModalityEncoder):
    """Encoder for text inputs."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int):
        """
        Initialize the text encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the word embeddings
            output_dim: Dimension of the output embedding
        """
        super().__init__(output_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the text encoder.
        
        Args:
            x: Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Encoded representation [batch_size, seq_len, output_dim]
        """
        embeddings = self.embedding(x)
        return self.projection(embeddings)


class ImageEncoder(ModalityEncoder):
    """Encoder for image inputs."""
    
    def __init__(self, input_channels: int, output_dim: int):
        """
        Initialize the image encoder.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            output_dim: Dimension of the output embedding
        """
        super().__init__(output_dim)
        
        # Simple CNN for image encoding
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Final projection
        self.fc = nn.Linear(128 * 4 * 4, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Encoded representation [batch_size, output_dim]
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AudioEncoder(ModalityEncoder):
    """Encoder for audio inputs."""
    
    def __init__(self, input_channels: int, output_dim: int):
        """
        Initialize the audio encoder.
        
        Args:
            input_channels: Number of input channels
            output_dim: Dimension of the output embedding
        """
        super().__init__(output_dim)
        
        # 1D CNN for audio encoding
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # Adaptive pooling to handle different input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # Final projection
        self.fc = nn.Linear(128 * 16, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        
        Args:
            x: Input audio [batch_size, channels, time]
            
        Returns:
            torch.Tensor: Encoded representation [batch_size, output_dim]
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MultimodalFusion(nn.Module):
    """Module for fusing embeddings from multiple modalities."""
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int):
        """
        Initialize the multimodal fusion module.
        
        Args:
            modality_dims: Dictionary mapping modality names to their embedding dimensions
            fusion_dim: Dimension of the fused embedding
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Projections for each modality
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modality_dims.items()
        })
        
        # Attention for weighted fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion module.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            
        Returns:
            torch.Tensor: Fused embedding
        """
        # Project each modality to the fusion dimension
        projected = {}
        for modality, embedding in embeddings.items():
            if modality in self.projections:
                projected[modality] = self.projections[modality](embedding)
        
        if not projected:
            raise ValueError("No valid modalities provided")
        
        # If only one modality, return its projection
        if len(projected) == 1:
            return next(iter(projected.values()))
        
        # Stack all projections
        stacked = torch.stack(list(projected.values()), dim=1)  # [batch_size, n_modalities, fusion_dim]
        
        # Compute attention weights
        attn_weights = self.attention(stacked).softmax(dim=1)  # [batch_size, n_modalities, 1]
        
        # Weighted sum
        fused = (stacked * attn_weights).sum(dim=1)  # [batch_size, fusion_dim]
        
        return fused