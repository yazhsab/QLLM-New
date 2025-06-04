"""
Training pipeline for multimodal quantum LLM.

This module provides utilities for training multimodal quantum language models
with different types of input data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm
import os
import json

from Qynthra.multimodal.preprocessing import MultimodalPreprocessor


class MultimodalDataset(Dataset):
    """Dataset for multimodal training."""
    
    def __init__(self, 
                 data: List[Dict[str, Any]],
                 preprocessor: MultimodalPreprocessor,
                 modalities: List[str] = ["text"],
                 max_samples: Optional[int] = None):
        """
        Initialize the multimodal dataset.
        
        Args:
            data: List of data samples, each containing modality inputs and labels
            preprocessor: Preprocessor for multimodal inputs
            modalities: List of modalities to include
            max_samples: Maximum number of samples to use
        """
        self.modalities = modalities
        self.preprocessor = preprocessor
        
        # Filter and limit data
        if max_samples is not None and max_samples < len(data):
            self.data = data[:max_samples]
        else:
            self.data = data
    
    def __len__(self) -> int:
        """Get the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (inputs, labels)
        """
        sample = self.data[idx]
        
        # Extract inputs for each modality
        inputs = {}
        for modality in self.modalities:
            if modality in sample:
                inputs[modality] = sample[modality]
        
        # Preprocess inputs
        processed_inputs = self.preprocessor(inputs)
        
        # Get labels
        if 'labels' in sample:
            labels = torch.tensor(sample['labels'], dtype=torch.long)
        else:
            # If no labels, use input text shifted by one position
            if 'text' in processed_inputs:
                text_input = processed_inputs['text']
                labels = torch.cat([text_input[:, 1:], torch.zeros((text_input.shape[0], 1), dtype=torch.long)], dim=1)
            else:
                raise ValueError("No labels provided and no text input for default labels")
        
        return processed_inputs, labels


class MultimodalTrainer:
    """Trainer for multimodal quantum LLM."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 device: str = "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model: Multimodal quantum LLM model
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': []
        }
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              epochs: int = 5,
              gradient_accumulation_steps: int = 1,
              max_grad_norm: float = 1.0,
              log_interval: int = 10,
              save_path: Optional[str] = None,
              callback: Optional[Callable] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Interval for logging
            save_path: Path to save model checkpoints
            callback: Callback function
            
        Returns:
            dict: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Create save directory if needed
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            train_steps = 0
            
            # Progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for step, (inputs, labels) in enumerate(pbar):
                # Move inputs and labels to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                
                # Calculate loss
                loss = self._compute_loss(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if needed
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                # Update metrics
                train_loss += loss.item() * gradient_accumulation_steps
                train_steps += 1
                
                # Update progress bar
                if step % log_interval == 0:
                    pbar.set_postfix({
                        'loss': train_loss / train_steps
                    })
            
            # Calculate epoch metrics
            train_loss = train_loss / train_steps
            train_perplexity = np.exp(train_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(train_perplexity)
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_perplexity = self.evaluate(val_dataloader)
                
                # Update history
                self.history['val_loss'].append(val_loss)
                self.history['val_perplexity'].append(val_perplexity)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {train_loss:.4f}, Perplexity = {train_perplexity:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {train_loss:.4f}, Perplexity = {train_perplexity:.4f}")
            
            # Save checkpoint
            if save_path is not None:
                checkpoint_path = f"{save_path}.epoch{epoch+1}"
                self.save_checkpoint(checkpoint_path)
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.model, self.history)
        
        # Save final model
        if save_path is not None:
            self.save_checkpoint(save_path)
        
        return self.history
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Data loader
            
        Returns:
            tuple: (loss, perplexity)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        # Disable gradient computation
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                # Move inputs and labels to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                
                # Calculate loss
                loss = self._compute_loss(logits, labels)
                
                # Update metrics
                total_loss += loss.item()
                total_steps += 1
        
        # Calculate metrics
        avg_loss = total_loss / total_steps
        perplexity = np.exp(avg_loss)
        
        # Set model back to training mode
        self.model.train()
        
        return avg_loss, perplexity
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Calculate cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, labels_flat)
        
        return loss
    
    def save_checkpoint(self, path: str):
        """
        Save a checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(f"{path}.model")
        
        # Save optimizer and scheduler states
        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, f"{path}.trainer")
    
    def load_checkpoint(self, path: str):
        """
        Load a checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        # Load model
        self.model.load(f"{path}.model")
        
        # Load optimizer and scheduler states
        checkpoint = torch.load(f"{path}.trainer")
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])


class MultimodalEvaluator:
    """Evaluator for multimodal quantum LLM."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the evaluator.
        
        Args:
            model: Multimodal quantum LLM model
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_text_generation(self, 
                                inputs: Dict[str, torch.Tensor],
                                tokenizer: Any,
                                max_length: int = 50,
                                temperature: float = 1.0,
                                top_k: Optional[int] = None,
                                top_p: Optional[float] = None,
                                num_samples: int = 1) -> List[str]:
        """
        Evaluate text generation.
        
        Args:
            inputs: Dictionary mapping modality names to their inputs
            tokenizer: Tokenizer for decoding
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            num_samples: Number of samples to generate
            
        Returns:
            list: Generated texts
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate multiple samples
        all_texts = []
        
        for _ in range(num_samples):
            # Generate tokens
            generated_ids = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Decode tokens
            for ids in generated_ids:
                text = tokenizer.decode(ids.tolist())
                all_texts.append(text)
        
        return all_texts
    
    def evaluate_metrics(self, 
                         dataloader: DataLoader,
                         metrics: List[Callable]) -> Dict[str, float]:
        """
        Evaluate metrics.
        
        Args:
            dataloader: Data loader
            metrics: List of metric functions
            
        Returns:
            dict: Metrics results
        """
        results = {metric.__name__: 0.0 for metric in metrics}
        total_samples = 0
        
        # Disable gradient computation
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating metrics"):
                # Move inputs and labels to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                
                # Calculate metrics
                batch_size = labels.shape[0]
                for metric in metrics:
                    results[metric.__name__] += metric(logits, labels) * batch_size
                
                total_samples += batch_size
        
        # Average metrics
        for metric_name in results:
            results[metric_name] /= total_samples
        
        return results


# Common evaluation metrics
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate accuracy.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        
    Returns:
        float: Accuracy
    """
    # Get predictions
    preds = logits.argmax(dim=-1)
    
    # Calculate accuracy
    correct = (preds == labels).float()
    
    return correct.mean().item()


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate perplexity.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        
    Returns:
        float: Perplexity
    """
    # Reshape for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Calculate cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits_flat, labels_flat)
    
    return np.exp(loss.item())