"""
Key-Value cache for multimodal quantum LLM.

This module provides an extension of the KV cache mechanism for multimodal inputs,
allowing efficient inference with multiple input modalities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class MultimodalKVCache:
    """Key-Value cache for multimodal quantum LLM."""
    
    def __init__(self, supported_modalities: List[str]):
        """
        Initialize the multimodal KV cache.
        
        Args:
            supported_modalities: List of supported modalities
        """
        self.supported_modalities = supported_modalities
        self.reset()
    
    def reset(self):
        """Reset the cache."""
        self.last_seq_len = 0
        self.quantum_states = []
        
        # Modality-specific caches
        self.modality_caches = {
            modality: {
                'last_seq_len': 0,
                'states': []
            }
            for modality in self.supported_modalities
        }
        
        # Cross-modal attention cache
        self.cross_modal_cache = {
            'last_update': 0,
            'states': []
        }
    
    def update(self, 
               seq_idx: int, 
               modality: str, 
               state: np.ndarray):
        """
        Update the cache for a specific modality and sequence position.
        
        Args:
            seq_idx: Sequence index
            modality: Modality name
            state: Quantum state
        """
        if modality not in self.supported_modalities:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Update modality-specific cache
        modality_cache = self.modality_caches[modality]
        abs_pos = modality_cache['last_seq_len'] + seq_idx
        
        if len(modality_cache['states']) <= abs_pos:
            modality_cache['states'].append(state)
        else:
            modality_cache['states'][abs_pos] = state
    
    def update_cross_modal(self, state: np.ndarray):
        """
        Update the cross-modal attention cache.
        
        Args:
            state: Cross-modal attention state
        """
        self.cross_modal_cache['states'].append(state)
        self.cross_modal_cache['last_update'] += 1
    
    def get_modality_state(self, modality: str, seq_idx: int) -> Optional[np.ndarray]:
        """
        Get the cached state for a specific modality and sequence position.
        
        Args:
            modality: Modality name
            seq_idx: Sequence index
            
        Returns:
            np.ndarray or None: Cached state if available
        """
        if modality not in self.supported_modalities:
            return None
        
        modality_cache = self.modality_caches[modality]
        abs_pos = modality_cache['last_seq_len'] + seq_idx
        
        if abs_pos < len(modality_cache['states']):
            return modality_cache['states'][abs_pos]
        
        return None
    
    def get_cross_modal_state(self, idx: int = -1) -> Optional[np.ndarray]:
        """
        Get the cached cross-modal attention state.
        
        Args:
            idx: Index of the state to get (-1 for the latest)
            
        Returns:
            np.ndarray or None: Cached state if available
        """
        if not self.cross_modal_cache['states']:
            return None
        
        if idx < 0:
            idx = len(self.cross_modal_cache['states']) + idx
        
        if 0 <= idx < len(self.cross_modal_cache['states']):
            return self.cross_modal_cache['states'][idx]
        
        return None
    
    def update_seq_len(self, modality: str, seq_len: int):
        """
        Update the sequence length for a specific modality.
        
        Args:
            modality: Modality name
            seq_len: Sequence length to add
        """
        if modality not in self.supported_modalities:
            return
        
        self.modality_caches[modality]['last_seq_len'] += seq_len
    
    def to_dict(self) -> Dict:
        """
        Convert the cache to a dictionary.
        
        Returns:
            dict: Cache as a dictionary
        """
        return {
            'last_seq_len': self.last_seq_len,
            'quantum_states': self.quantum_states,
            'modality_caches': self.modality_caches,
            'cross_modal_cache': self.cross_modal_cache
        }
    
    @classmethod
    def from_dict(cls, cache_dict: Dict, supported_modalities: List[str]) -> 'MultimodalKVCache':
        """
        Create a cache from a dictionary.
        
        Args:
            cache_dict: Dictionary representation of the cache
            supported_modalities: List of supported modalities
            
        Returns:
            MultimodalKVCache: Cache instance
        """
        cache = cls(supported_modalities)
        
        cache.last_seq_len = cache_dict.get('last_seq_len', 0)
        cache.quantum_states = cache_dict.get('quantum_states', [])
        
        # Update modality caches
        for modality, modality_cache in cache_dict.get('modality_caches', {}).items():
            if modality in supported_modalities:
                cache.modality_caches[modality] = modality_cache
        
        # Update cross-modal cache
        cache.cross_modal_cache = cache_dict.get('cross_modal_cache', {
            'last_update': 0,
            'states': []
        })
        
        return cache


class MultimodalQLLMWithKVCache:
    """Mixin class for adding KV cache to multimodal QLLM."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the KV cache."""
        super().__init__(*args, **kwargs)
        self.kv_cache = None
    
    def reset_cache(self):
        """Reset the KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.reset()
        else:
            self.kv_cache = MultimodalKVCache(self.supported_modalities)
    
    def forward_with_cache(self, 
                          inputs: Dict[str, torch.Tensor], 
                          use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with KV cache support.
        
        Args:
            inputs: Dictionary mapping modality names to their inputs
            use_cache: Whether to use and update KV cache
            
        Returns:
            tuple: (output logits, updated KV cache)
        """
        # Initialize or get KV cache
        if use_cache and self.kv_cache is None:
            self.kv_cache = MultimodalKVCache(self.supported_modalities)
        
        # Process each modality with its encoder
        modality_embeddings = {}
        
        for modality, modality_input in inputs.items():
            if modality in self.modality_encoders:
                modality_embeddings[modality] = self.modality_encoders[modality](modality_input)
        
        # Get batch size and sequence length from text input (if available)
        if 'text' in inputs:
            batch_size, seq_len = inputs['text'].shape
        else:
            # For non-text inputs, assume batch size from first modality and seq_len=1
            batch_size = next(iter(inputs.values())).shape[0]
            seq_len = 1
        
        # Process each sequence position with quantum circuits
        outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            
            for s in range(seq_len):
                # Prepare quantum inputs for each modality
                quantum_inputs = {}
                
                for modality, embedding in modality_embeddings.items():
                    # Get embedding for this position and batch
                    if modality == 'text':
                        # For text, we have sequence dimension
                        emb = embedding[b, s].detach().numpy()
                    else:
                        # For other modalities, we might not have sequence dimension
                        emb = embedding[b].detach().numpy()
                    
                    # Normalize to [0, 2π]
                    emb = 2 * np.pi * (emb - np.min(emb)) / (np.max(emb) - np.min(emb) + 1e-8)
                    
                    quantum_inputs[modality] = emb
                
                # Get absolute position for cache
                abs_pos = self.kv_cache.modality_caches['text']['last_seq_len'] + s if use_cache else s
                
                # Check if we have a cached state
                cached_state = None
                if use_cache:
                    cached_state = self.kv_cache.get_modality_state('text', s)
                
                if cached_state is not None:
                    # Use cached state
                    result = cached_state
                else:
                    # Create and run quantum circuit for this position
                    circuit = self.quantum_circuit_fn(abs_pos)
                    result = circuit(quantum_inputs, self.params)
                    
                    # Update cache if using cache
                    if use_cache:
                        self.kv_cache.update(s, 'text', result)
                
                # Convert to tensor
                result_tensor = torch.tensor(result, dtype=torch.float32)
                
                # Project to vocabulary size using classical layer
                logits = self.output_projection(result_tensor)
                
                seq_outputs.append(logits)
            
            outputs.append(torch.stack(seq_outputs))
        
        # Update cache sequence length
        if use_cache:
            self.kv_cache.update_seq_len('text', seq_len)
        
        return torch.stack(outputs), self.kv_cache if use_cache else None
    
    def generate_with_cache(self, 
                           inputs: Dict[str, torch.Tensor], 
                           max_length: int = 50, 
                           temperature: float = 1.0,
                           top_k: Optional[int] = None,
                           top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text with KV cache for efficiency.
        
        Args:
            inputs: Dictionary mapping modality names to their inputs
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        self.eval()
        
        # Reset cache
        self.reset_cache()
        
        # Get batch size from text input (if available)
        if 'text' in inputs:
            batch_size, seq_len = inputs['text'].shape
        else:
            # For non-text inputs, assume batch size from first modality
            batch_size = next(iter(inputs.values())).shape[0]
            seq_len = 1
        
        # Initialize with input_ids
        if 'text' in inputs:
            generated = inputs['text'].clone()
        else:
            # If no text input, start with empty sequence
            generated = torch.zeros((batch_size, 0), dtype=torch.long)
        
        # Process initial input
        logits, _ = self.forward_with_cache(inputs, use_cache=True)
        
        # Generate tokens
        for i in range(max_length - generated.shape[1]):
            # Get predictions for the last token
            last_token_logits = logits[:, -1, :]
            
            # Apply temperature
            last_token_logits = last_token_logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = torch.topk(last_token_logits, k=top_k, dim=-1)[0][:, -1].unsqueeze(-1)
                last_token_logits[last_token_logits < indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    last_token_logits[b, indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Prepare inputs for next iteration
            next_inputs = {
                'text': next_token
            }
            
            # Forward pass with cache
            logits, _ = self.forward_with_cache(next_inputs, use_cache=True)
        
        return generated