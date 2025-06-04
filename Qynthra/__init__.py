"""
Qynthra package.

This package provides a quantum-enhanced large language model implementation
using PennyLane for quantum computing integration.
"""

from Qynthra.qllm_base import QLLMBase
from Qynthra.qllm_advanced import QLLMAdvanced, QLLMWithKVCache
from Qynthra.tokenization import QLLMTokenizer

# Import multimodal components
from Qynthra.multimodal.model import MultimodalQLLM
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
    audio_to_quantum_encoding
)
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
    MultimodalEvaluator
)

__version__ = "0.1.0"