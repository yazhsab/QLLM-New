"""
Preprocessing utilities for multimodal inputs.

This module provides functions for preprocessing different types of inputs
(text, images, audio) before feeding them into the multimodal QLLM.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import PIL.Image as Image
from PIL import ImageOps
import librosa
import torchvision.transforms as transforms


class TextPreprocessor:
    """Preprocessor for text inputs."""
    
    def __init__(self, tokenizer: Any, max_length: int = 128):
        """
        Initialize the text preprocessor.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Preprocess text input.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            torch.Tensor: Tokenized text [batch_size, seq_len]
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        token_ids = []
        for t in text:
            ids = self.tokenizer.encode(t)
            
            # Pad or truncate to max_length
            if len(ids) < self.max_length:
                ids = ids + [0] * (self.max_length - len(ids))
            else:
                ids = ids[:self.max_length]
                
            token_ids.append(ids)
        
        return torch.tensor(token_ids)


class ImagePreprocessor:
    """Preprocessor for image inputs."""
    
    def __init__(self, 
                 size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize the image preprocessor.
        
        Args:
            size: Size to resize images to
            normalize: Whether to normalize images
            mean: Mean for normalization
            std: Standard deviation for normalization
        """
        self.size = size
        
        # Create transform pipeline
        transforms_list = [
            transforms.Resize(size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transforms_list)
    
    def __call__(self, images: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image input.
        
        Args:
            images: Input image or list of images
            
        Returns:
            torch.Tensor: Processed images [batch_size, channels, height, width]
        """
        # Handle different input types
        if isinstance(images, Image.Image):
            # Single PIL image
            return self.transform(images).unsqueeze(0)
        
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            # List of PIL images
            return torch.stack([self.transform(img) for img in images])
        
        elif isinstance(images, np.ndarray):
            # NumPy array
            if images.ndim == 3:
                # Single image
                pil_image = Image.fromarray(images.astype(np.uint8))
                return self.transform(pil_image).unsqueeze(0)
            elif images.ndim == 4:
                # Batch of images
                return torch.stack([
                    self.transform(Image.fromarray(img.astype(np.uint8)))
                    for img in images
                ])
        
        elif isinstance(images, torch.Tensor):
            # PyTorch tensor
            if images.ndim == 3:
                # Single image [channels, height, width]
                return images.unsqueeze(0)
            elif images.ndim == 4:
                # Batch of images [batch_size, channels, height, width]
                return images
        
        raise ValueError(f"Unsupported image input type: {type(images)}")


class AudioPreprocessor:
    """Preprocessor for audio inputs."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 duration: Optional[float] = None):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            duration: Duration to truncate/pad to (in seconds)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
    
    def __call__(self, audio: Union[np.ndarray, List[np.ndarray], torch.Tensor]) -> torch.Tensor:
        """
        Preprocess audio input.
        
        Args:
            audio: Input audio or list of audio signals
            
        Returns:
            torch.Tensor: Processed audio spectrograms [batch_size, 1, n_mels, time]
        """
        # Handle different input types
        if isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                # Single audio signal
                return self._process_single_audio(audio).unsqueeze(0)
            elif audio.ndim == 2:
                # Batch of audio signals
                return torch.stack([
                    self._process_single_audio(signal) for signal in audio
                ])
        
        elif isinstance(audio, list) and all(isinstance(a, np.ndarray) for a in audio):
            # List of audio signals
            return torch.stack([
                self._process_single_audio(signal) for signal in audio
            ])
        
        elif isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                # Single audio signal
                return self._process_single_audio(audio.numpy()).unsqueeze(0)
            elif audio.ndim == 2:
                # Batch of audio signals
                return torch.stack([
                    self._process_single_audio(signal.numpy()) for signal in audio
                ])
            elif audio.ndim == 3 and audio.shape[1] == 1:
                # Already processed spectrograms [batch_size, 1, time]
                return audio
        
        raise ValueError(f"Unsupported audio input type: {type(audio)}")
    
    def _process_single_audio(self, signal: np.ndarray) -> torch.Tensor:
        """
        Process a single audio signal.
        
        Args:
            signal: Audio signal
            
        Returns:
            torch.Tensor: Processed audio spectrogram [1, n_mels, time]
        """
        # Resample if needed
        if self.sample_rate is not None:
            signal = librosa.resample(signal, orig_sr=self.sample_rate, target_sr=self.sample_rate)
        
        # Truncate or pad to duration
        if self.duration is not None:
            target_length = int(self.sample_rate * self.duration)
            if len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))
            else:
                signal = signal[:target_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Convert to tensor
        return torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)


class MultimodalPreprocessor:
    """Preprocessor for multimodal inputs."""
    
    def __init__(self, 
                 tokenizer: Any,
                 text_max_length: int = 128,
                 image_size: Tuple[int, int] = (224, 224),
                 audio_sample_rate: int = 16000,
                 audio_duration: Optional[float] = None):
        """
        Initialize the multimodal preprocessor.
        
        Args:
            tokenizer: Tokenizer for text processing
            text_max_length: Maximum text sequence length
            image_size: Size to resize images to
            audio_sample_rate: Target audio sample rate
            audio_duration: Duration to truncate/pad audio to
        """
        self.text_processor = TextPreprocessor(tokenizer, text_max_length)
        self.image_processor = ImagePreprocessor(image_size)
        self.audio_processor = AudioPreprocessor(audio_sample_rate, duration=audio_duration)
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess multimodal inputs.
        
        Args:
            inputs: Dictionary mapping modality names to their inputs
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed inputs
        """
        outputs = {}
        
        if 'text' in inputs:
            outputs['text'] = self.text_processor(inputs['text'])
        
        if 'image' in inputs:
            outputs['image'] = self.image_processor(inputs['image'])
        
        if 'audio' in inputs:
            outputs['audio'] = self.audio_processor(inputs['audio'])
        
        return outputs