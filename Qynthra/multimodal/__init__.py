"""
Multimodal support for Quantum LLM.

This module provides functionality for handling multiple input modalities
(text, images, audio) in the Quantum LLM framework.
"""

from Qynthra.multimodal.encoders import (
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    MultimodalFusion
)

from Qynthra.multimodal.data_encoding import (
    create_multimodal_embedding_circuit,
    create_modality_specific_embedding,
    image_to_quantum_encoding,
    audio_to_quantum_encoding,
    quantum_multimodal_fusion
)

from Qynthra.multimodal.model import MultimodalQLLM

from Qynthra.multimodal.attention import (
    ClassicalCrossModalAttention,
    quantum_cross_modal_attention
)

from Qynthra.multimodal.preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor,
    MultimodalPreprocessor
)

from Qynthra.multimodal.kv_cache import (
    MultimodalKVCache,
    MultimodalQLLMWithKVCache
)

from Qynthra.multimodal.training import (
    MultimodalDataset,
    MultimodalTrainer,
    MultimodalEvaluator,
    accuracy,
    perplexity
)