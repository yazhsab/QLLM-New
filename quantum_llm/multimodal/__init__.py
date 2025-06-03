"""
Multimodal support for Quantum LLM.

This module provides functionality for handling multiple input modalities
(text, images, audio) in the Quantum LLM framework.
"""

from quantum_llm.multimodal.encoders import (
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    MultimodalFusion
)

from quantum_llm.multimodal.data_encoding import (
    create_multimodal_embedding_circuit,
    create_modality_specific_embedding,
    image_to_quantum_encoding,
    audio_to_quantum_encoding,
    quantum_multimodal_fusion
)

from quantum_llm.multimodal.model import MultimodalQLLM

from quantum_llm.multimodal.attention import (
    ClassicalCrossModalAttention,
    quantum_cross_modal_attention
)

from quantum_llm.multimodal.preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor,
    MultimodalPreprocessor
)

from quantum_llm.multimodal.kv_cache import (
    MultimodalKVCache,
    MultimodalQLLMWithKVCache
)

from quantum_llm.multimodal.training import (
    MultimodalDataset,
    MultimodalTrainer,
    MultimodalEvaluator,
    accuracy,
    perplexity
)