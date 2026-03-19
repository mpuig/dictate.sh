# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx>=0.22.0",
#     "mlx-lm>=0.22.0",
#     "numpy",
#     "sounddevice",
#     "setuptools",
#     "transformers",
#     "huggingface_hub",
#     "webrtcvad",
#     "rich",
# ]
# ///
"""
Standalone, low-latency transcription for Apple Silicon.

Uses MLX so inference stays local and fast on macOS.
VAD marks turn boundaries to avoid constant analysis while keeping responses timely.

Design notes:
    - Rolling-window ASR trades some stability for low latency.
    - VAD reduces GPU work by running ASR only when useful.
    - MLX is serialized to avoid concurrency issues on the GPU.
    - Logging goes to stderr so transcripts stay clean on stdout.

Troubleshooting:
    - If turns split too often: increase --vad-silence-ms or lower --vad-mode.
    - If nothing transcribes: check mic permissions or run --list-devices.
    - If output feels laggy: reduce --transcribe-interval.

Usage:
    uv run stt.py
    uv run stt.py --model mlx-community/Qwen3-ASR-1.7B-8bit
    uv run stt.py --analyze  # opt-in if you want intent analysis
    uv run stt.py --vad-mode 3 --vad-silence-ms 700

Models (MLX Qwen3-ASR):
    - mlx-community/Qwen3-ASR-0.6B-4bit: fastest, lowest quality.
    - mlx-community/Qwen3-ASR-0.6B-8bit: good balance (default).
    - mlx-community/Qwen3-ASR-0.6B-bf16: higher quality, more RAM.
    - mlx-community/Qwen3-ASR-1.7B-8bit: higher quality, slower.

LLM models for --llm-model (examples; availability may change):
    - mlx-community/Qwen3-0.6B-4bit: fastest, lowest RAM (default).
    - mlx-community/Qwen3-1.7B-4bit: better quality, slower.
    - mlx-community/Mistral-7B-Instruct-v0.2-4bit: heavier.
    - mlx-community/Llama-3.1-8B-Instruct-4bit: heavier.
"""

import argparse
import asyncio
import contextlib
import glob
import io
import json
import logging
import math
import os  # noqa: E402
import re
import shutil
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Protocol, Sequence, Tuple

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Keep this before MLX/Transformers imports to reduce noisy warnings and progress
# bars. The # noqa: E402 imports below are intentional for this reason.

# Reduce noisy stderr in a real-time CLI.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# MLX deprecation chatter overwhelms live output.
import mlx.core as mx  # noqa: E402

if hasattr(mx, "set_warnings_enabled"):
    mx.set_warnings_enabled(False)
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import webrtcvad_compat as webrtcvad  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.logging import RichHandler  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"
DEFAULT_LLM_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_LANGUAGE = "English"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TRANSCRIBE_INTERVAL = 0.5
DEFAULT_VAD_FRAME_MS = 30
DEFAULT_VAD_MODE = 2
DEFAULT_VAD_SILENCE_MS = 500
DEFAULT_MIN_WORDS = 3
DEFAULT_MAX_BUFFER_SECONDS = 30
DEFAULT_AUDIO_QUEUE_MAXSIZE = 200

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("speech")


@contextlib.contextmanager
def _suppress_output():
    """Hide noisy library prints/warnings during background inference."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
        ):
            yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull)


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class TokenizerLike(Protocol):
    def encode(self, text: str, return_tensors: str) -> Any: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def apply_chat_template(
            self,
            messages: Sequence[Dict[str, str]],
            tokenize: bool,
            add_generation_prompt: bool,
    ) -> str: ...


class FeatureExtractorLike(Protocol):
    def __call__(
            self,
            audio: np.ndarray,
            sampling_rate: int,
            return_attention_mask: bool,
            truncation: bool,
            padding: bool,
            return_tensors: str,
    ) -> Dict[str, Any]: ...


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    scale_embedding: bool = False
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480


@dataclass
class TextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0


@dataclass
class ModelConfig:
    audio_config: AudioEncoderConfig = None
    text_config: TextConfig = None
    audio_token_id: int = 151676
    support_languages: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig(
                **{
                    k: v
                    for k, v in self.audio_config.items()
                    if k in AudioEncoderConfig.__dataclass_fields__
                }
            )
        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig(
                **{
                    k: v
                    for k, v in self.text_config.items()
                    if k in TextConfig.__dataclass_fields__
                }
            )


# =============================================================================
# Model Architecture
# =============================================================================


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Return an additive causal mask to prevent attention to future tokens."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def _floor_div(a: mx.array, b: int) -> mx.array:
    """Floor-divide while keeping MLX tensors, avoiding host/device sync."""
    return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def _get_feat_extract_output_lengths(input_lengths: mx.array) -> mx.array:
    """Track time-downsampling so chunk masks align with conv output."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = _floor_div(input_lengths_leave - 1, 2) + 1
    output_lengths = (
            _floor_div(_floor_div(feat_lengths - 1, 2) + 1 - 1, 2)
            + 1
            + (input_lengths // 100) * 13
    )
    return output_lengths


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed positions so timing is known without extra learned parameters."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32)
        )
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        scaled_time = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], axis=1
        )

    def __call__(self, seqlen: int) -> mx.array:
        return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    """Self-attention to relate distant audio frames for context."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
            self, hidden_states: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states) * self.scaling
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """Transformer block to mix local and global audio features."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
            self, hidden_states: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AudioEncoder(nn.Module):
    """Audio encoder that compresses time then builds contextual features."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer

        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv, embed_dim, bias=False
        )
        self.positional_embedding = SinusoidalPositionEmbedding(
            config.max_source_positions, embed_dim
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    def _create_block_attention_mask(
            self, seq_len: int, cu_seqlens: List[int], dtype: mx.Dtype
    ) -> mx.array:
        """Limit attention to chunk boundaries for stability and speed."""
        mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            mask[start:end, start:end] = 0.0
        return mask

    def _compute_chunk_layout(
            self, feature_lens: np.ndarray, chunk_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Define chunking so long inputs stay bounded in memory/latency."""
        chunk_counts = np.ceil(feature_lens / chunk_size).astype(np.int32)
        chunk_lengths: List[int] = []
        for feat_len, num_chunks in zip(feature_lens, chunk_counts):
            feat_len = int(feat_len)
            num_chunks = int(num_chunks)
            for j in range(num_chunks):
                if j == num_chunks - 1:
                    remainder = feat_len % chunk_size
                    chunk_lengths.append(chunk_size if remainder == 0 else remainder)
                else:
                    chunk_lengths.append(chunk_size)
        return chunk_counts, np.array(chunk_lengths, dtype=np.int32)

    def _slice_feature_chunks(
            self,
            input_features: mx.array,
            feature_lens: np.ndarray,
            chunk_counts: np.ndarray,
            chunk_size: int,
    ) -> List[mx.array]:
        """Cut features into chunks so conv/attention operate on windows."""
        chunks: List[mx.array] = []
        for feat, feat_len, num_chunks in zip(
                input_features, feature_lens, chunk_counts
        ):
            feat_len = int(feat_len)
            num_chunks = int(num_chunks)
            pos = 0
            remainder = feat_len % chunk_size
            for j in range(num_chunks):
                clen = (
                    chunk_size if (j < num_chunks - 1 or remainder == 0) else remainder
                )
                chunks.append(feat[:, pos: pos + clen])
                pos += clen
        return chunks

    def _pad_chunks(
            self, chunks: List[mx.array], chunk_lengths: np.ndarray
    ) -> Tuple[mx.array, int]:
        """Pad for batching so convs run as a single dense tensor."""
        max_chunk_len = int(chunk_lengths.max())
        padded_chunks: List[mx.array] = []
        for chunk, clen in zip(chunks, chunk_lengths):
            clen = int(clen)
            if clen < max_chunk_len:
                chunk = mx.pad(chunk, [(0, 0), (0, max_chunk_len - clen)])
            padded_chunks.append(chunk)
        return mx.stack(padded_chunks, axis=0), max_chunk_len

    def _build_cu_seqlens(
            self, aftercnn_lens: np.ndarray, window_aftercnn: int
    ) -> List[int]:
        """Provide segment boundaries so attention stays inside windows."""
        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens:
            cnn_len = int(cnn_len)
            full_windows = cnn_len // window_aftercnn
            if full_windows:
                cu_chunk_lens.extend([window_aftercnn] * full_windows)
            remainder = cnn_len % window_aftercnn
            if remainder:
                cu_chunk_lens.append(remainder)
        return np.cumsum(cu_chunk_lens).tolist()

    def __call__(
            self,
            input_features: mx.array,
            feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode audio features into sequence embeddings.

        Flow:
        - determine feature lengths from mask or input shape
        - chunk along time, pad to a common chunk length
        - run conv downsampling + projection + positional embeddings
        - build block attention mask for chunked layout
        - apply transformer layers + output projections
        """
        if feature_attention_mask is not None:
            feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
        else:
            feature_lens = mx.array(
                [input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32
            )

        feature_lens_np = np.array(feature_lens)
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_size = self.n_window * 2
        chunk_counts, chunk_lengths = self._compute_chunk_layout(
            feature_lens_np, chunk_size
        )
        chunks = self._slice_feature_chunks(
            input_features, feature_lens_np, chunk_counts, chunk_size
        )
        padded_feature, _ = self._pad_chunks(chunks, chunk_lengths)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(
            mx.array(chunk_lengths)
        )
        feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
        max_len_after_cnn = int(feature_lens_after_cnn_np.max())

        x = padded_feature[:, :, :, None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        b, f, t, c = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        x = self.conv_out(x)
        x = x + self.positional_embedding(x.shape[1])[None, :, :]

        hidden_list = [
            x[i, : int(feature_lens_after_cnn_np[i])] for i in range(x.shape[0])
        ]
        hidden_states = mx.concatenate(hidden_list, axis=0)

        aftercnn_lens_np = np.array(aftercnn_lens)
        window_aftercnn = max_len_after_cnn * (
                self.n_window_infer // (self.n_window * 2)
        )
        cu_seqlens = self._build_cu_seqlens(aftercnn_lens_np, window_aftercnn)
        attention_mask = self._create_block_attention_mask(
            hidden_states.shape[0], cu_seqlens, hidden_states.dtype
        )
        attention_mask = attention_mask[None, None, :, :]
        hidden_states = hidden_states[None, :, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = self.ln_post(hidden_states[0])
        hidden_states = nn.gelu(self.proj1(hidden_states))
        return self.proj2(hidden_states)


class TextAttention(nn.Module):
    """Self-attention so text tokens can condition on prior context."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
            self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            B, L, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache:
            keys, values = cache.update_and_fetch(keys, values)

        mask = create_additive_causal_mask(queries.shape[2], offset=offset).astype(
            queries.dtype
        )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(
            output.transpose(0, 2, 1, 3).reshape(B, -1, self.num_heads * self.head_dim)
        )


class TextMLP(nn.Module):
    """Nonlinear mixing to expand and compress token features."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """Decoder block to refine tokens with attention + MLP."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TextAttention(config, layer_idx)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
            self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cache=cache)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
            self,
            input_ids: Optional[mx.array] = None,
            inputs_embeds: Optional[mx.array] = None,
            cache: Optional[List[Any]] = None,
    ) -> mx.array:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )
        cache = cache or [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cache=cache[i])
        return self.norm(hidden_states)


class Qwen3ASRModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextModel(config.text_config)
        self.lm_head = (
            None
            if config.text_config.tie_word_embeddings
            else nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        )

    def get_audio_features(
            self,
            input_features: mx.array,
            feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.audio_tower(input_features, feature_attention_mask)

    def __call__(
            self,
            input_ids: mx.array,
            input_embeddings: Optional[mx.array] = None,
            input_features: Optional[mx.array] = None,
            feature_attention_mask: Optional[mx.array] = None,
            cache: Optional[List[Any]] = None,
    ) -> mx.array:
        inputs_embeds = (
            input_embeddings
            if input_embeddings is not None
            else self.model.embed_tokens(input_ids)
        )

        if input_features is not None and (
                cache is None or cache[0] is None or cache[0].offset == 0
        ):
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            ).astype(inputs_embeds.dtype)
            audio_token_mask = input_ids == self.config.audio_token_id

            if audio_token_mask.any():
                batch_size, seq_len, hidden_dim = inputs_embeds.shape
                flat_mask_np = np.array(audio_token_mask.reshape(-1))
                audio_indices = np.nonzero(flat_mask_np)[0]
                if len(audio_indices) > 0 and audio_features.shape[0] > 0:
                    num_to_replace = min(len(audio_indices), audio_features.shape[0])
                    flat_embeds = inputs_embeds.reshape(-1, hidden_dim)
                    indices = mx.array(audio_indices[:num_to_replace])
                    replacement = (
                        mx.zeros_like(flat_embeds)
                        .at[indices]
                        .add(audio_features[:num_to_replace])
                    )
                    mask = (
                        mx.zeros((flat_embeds.shape[0],), dtype=flat_embeds.dtype)
                        .at[indices]
                        .add(1)
                    )
                    flat_embeds = mx.where(mask[:, None] > 0, replacement, flat_embeds)
                    inputs_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_dim)

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)
        return (
            self.model.embed_tokens.as_linear(hidden_states)
            if self.lm_head is None
            else self.lm_head(hidden_states)
        )

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        return 16000

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        is_formatted = not any(k.startswith("thinker.") for k in weights.keys())
        for k, v in weights.items():
            if k.startswith("thinker."):
                k = k[len("thinker."):]
            if k == "lm_head.weight":
                continue
            if (
                    not is_formatted
                    and "conv2d" in k
                    and "weight" in k
                    and len(v.shape) == 4
            ):
                v = v.transpose(0, 2, 3, 1)
            sanitized[k] = v
        return sanitized


# =============================================================================
# Model Loading
# =============================================================================


def load_qwen3_asr(
        model_path: str,
) -> Tuple[Qwen3ASRModel, TokenizerLike, FeatureExtractorLike]:
    """Load aligned weights + preprocessing so inference stays consistent."""
    import os

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import transformers
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)

    from transformers import AutoTokenizer, WhisperFeatureExtractor

    # Ensure artifacts are local so loading is consistent and offline-ready.
    local_path = Path(model_path)
    if not local_path.exists():
        local_path = Path(
            snapshot_download(
                model_path,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            )
        )

    # Config drives architecture/quantization; keep as source of truth.
    with open(local_path / "config.json", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Support configs that wrap audio/text settings.
    if "thinker_config" in config_dict:
        thinker = config_dict["thinker_config"]
        config_dict["audio_config"] = thinker.get("audio_config", {})
        config_dict["text_config"] = thinker.get("text_config", {})
        config_dict["audio_token_id"] = thinker.get("audio_token_id", 151676)

    config = ModelConfig(
        audio_config=config_dict.get("audio_config"),
        text_config=config_dict.get("text_config"),
        audio_token_id=config_dict.get("audio_token_id", 151676),
        support_languages=config_dict.get("support_languages", []),
    )

    # Instantiate structure before loading weights.
    model = Qwen3ASRModel(config)

    # Load all shards before sanitizing for layout differences.
    weight_files = glob.glob(str(local_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    weights = Qwen3ASRModel.sanitize(weights)

    # Respect model-provided quantization to match weights.
    quantization = config_dict.get("quantization")
    if quantization:

        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            if p.startswith("audio_tower"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    # Match preprocessing to the model artifacts.
    prev_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path), trust_remote_code=True
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(str(local_path))
    finally:
        transformers.logging.set_verbosity(prev_verbosity)

    return model, tokenizer, feature_extractor


# =============================================================================
# Transcription
# =============================================================================


def transcribe(
        model: Qwen3ASRModel,
        tokenizer: TokenizerLike,
        feature_extractor: FeatureExtractorLike,
        audio: np.ndarray,
        language: str = "English",
        max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """Stream tokens to keep transcription latency low."""
    from mlx_lm.generate import generate_step

    # Match the model's expected feature pipeline.
    audio_inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    input_features = mx.array(audio_inputs["input_features"])
    feature_attention_mask = mx.array(audio_inputs["attention_mask"])

    # Needed to size the audio pad tokens in the prompt.
    audio_lengths = feature_attention_mask.sum(axis=-1)
    aftercnn_lens = _get_feat_extract_output_lengths(audio_lengths)
    num_audio_tokens = int(aftercnn_lens[0].item())

    # Qwen3-ASR expects audio pads inside the chat template.
    supported = model.config.support_languages or []
    supported_lower = {lang.lower(): lang for lang in supported}
    lang_name = supported_lower.get(language.lower(), language)

    prompt = (
        f"<|im_start|>system\n<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {lang_name}<asr_text>"
    )
    input_ids = mx.array(tokenizer.encode(prompt, return_tensors="np"))

    # Compute audio features once for embedding replacement.
    audio_features = model.get_audio_features(input_features, feature_attention_mask)
    mx.eval(audio_features)

    # Replace audio token embeddings with audio features.
    inputs_embeds = model.model.embed_tokens(input_ids)
    audio_features = audio_features.astype(inputs_embeds.dtype)
    audio_token_mask = input_ids == model.config.audio_token_id

    if audio_token_mask.any():
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        flat_mask_np = np.array(audio_token_mask.reshape(-1))
        audio_indices = np.nonzero(flat_mask_np)[0]

        if len(audio_indices) > 0:
            num_to_replace = min(len(audio_indices), audio_features.shape[0])
            flat_embeds = inputs_embeds.reshape(-1, hidden_dim)
            indices = mx.array(audio_indices[:num_to_replace])
            replacement = (
                mx.zeros_like(flat_embeds)
                .at[indices]
                .add(audio_features[:num_to_replace])
            )
            mask = (
                mx.zeros((flat_embeds.shape[0],), dtype=flat_embeds.dtype)
                .at[indices]
                .add(1)
            )
            flat_embeds = mx.where(mask[:, None] > 0, replacement, flat_embeds)
            inputs_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_dim)

    mx.eval(inputs_embeds)
    input_embeddings = inputs_embeds[0]
    prompt_ids = input_ids[0] if input_ids.ndim > 1 else input_ids

    eos_token_ids = [151645, 151643]

    for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=input_embeddings,
            model=model,
            max_tokens=max_tokens,
    ):
        if token in eos_token_ids:
            break
        yield tokenizer.decode([int(token)])


# =============================================================================
# Real-time Transcriber
# =============================================================================


INTENT_EXPLAIN_PROMPT = """Analyze this speech and respond with exactly 3 lines:
INTENT: <primary intent in 2-3 words>
ENTITIES: <key items or names, comma-separated, or "none">
ACTION: <what should happen next, one short sentence>

Speech: "{text}" /no_think"""


class RealtimeTranscriber:
    """Encapsulates the async pipeline so capture, VAD, and ASR stay coordinated."""

    def __init__(
            self,
            model_path: str = DEFAULT_ASR_MODEL,
            language: str = DEFAULT_LANGUAGE,
            transcribe_interval: float = DEFAULT_TRANSCRIBE_INTERVAL,
            vad_frame_ms: int = DEFAULT_VAD_FRAME_MS,
            vad_mode: int = DEFAULT_VAD_MODE,
            vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
            min_words: int = DEFAULT_MIN_WORDS,
            analyze: bool = False,
            llm_model: Optional[str] = None,
            device: Optional[int] = None,
            no_ui: bool = False,
    ):
        self.model_path = model_path
        self.language = language
        self.transcribe_interval = transcribe_interval
        self.vad_frame_ms = vad_frame_ms
        self.vad_mode = vad_mode
        self.vad_silence_ms = vad_silence_ms
        self.min_words = min_words
        self.analyze = analyze
        self.llm_model_name = llm_model or DEFAULT_LLM_MODEL
        self.device = device
        self.no_ui = no_ui

        self.sample_rate = DEFAULT_SAMPLE_RATE
        # Bound RAM and latency by limiting the rolling window.
        self.max_buffer_seconds = DEFAULT_MAX_BUFFER_SECONDS

        self.audio_queue = asyncio.Queue(maxsize=DEFAULT_AUDIO_QUEUE_MAXSIZE)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.model = None
        self.tokenizer: Optional[TokenizerLike] = None
        self.feature_extractor: Optional[FeatureExtractorLike] = None
        self.llm = None
        self.llm_tokenizer: Optional[TokenizerLike] = None

        # Ring buffer avoids reallocations; int16 matches the input stream.
        self.max_buffer_samples = int(self.max_buffer_seconds * self.sample_rate)
        self.audio_buffer = np.zeros(self.max_buffer_samples, dtype=np.int16)
        self.buffer_write_pos = 0
        self.buffer_filled = 0
        self.total_samples_written = 0
        self.last_transcribed_sample = 0
        self.buffer_lock = asyncio.Lock()

        # MLX is not re-entrant; serialize GPU work to avoid races.
        self.gpu_lock = asyncio.Lock()

        # Track speech boundaries for turn detection.
        if self.vad_frame_ms not in (10, 20, 30):
            raise ValueError("vad_frame_ms must be one of: 10, 20, 30")
        if not (0 <= self.vad_mode <= 3):
            raise ValueError("vad_mode must be between 0 and 3")
        self.vad = webrtcvad.Vad(self.vad_mode)
        self.vad_frame_samples = int(self.sample_rate * self.vad_frame_ms / 1000)
        self.vad_silence_frames = int(
            math.ceil(self.vad_silence_ms / self.vad_frame_ms)
        )
        self.vad_residual = np.array([], dtype=np.int16)
        self.vad_speech_detected = False
        self.vad_silence_count = 0
        self.frame_size = self.vad_frame_samples

        # Track output across updates.
        self.current_transcript = ""
        self.last_transcript = ""
        self.turn_complete = False
        self.pending_analysis = None  # Defer display until analysis is finished.

        # Rich UI state (stderr) + clean transcript output (stdout).
        self.console_out = Console()
        self.console_ui = Console(stderr=True, force_terminal=True)
        self.live: Optional[Live] = None
        self.ui_status = "Starting"
        self.ui_partial = ""
        self.ui_history: List[Tuple[str, Optional[Dict[str, str]]]] = []
        self.ui_max_history = 50
        self.ui_vad_state = "silence"
        self.ui_buffer_seconds = 0.0
        self.ui_queue_size = 0
        self.ui_asr_ms: Optional[float] = None
        self.ui_analysis_ms: Optional[float] = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Keep callback lightweight by deferring work to the async loop."""
        data = indata.reshape(-1).copy()
        self.loop.call_soon_threadsafe(
            lambda: (
                self.audio_queue.put_nowait(data)
                if not self.audio_queue.full()
                else None
            )
        )

    def _short_model_name(self, name: Optional[str]) -> str:
        """Render a short model name for UI labels."""
        if not name:
            return "--"
        return name.split("/")[-1]

    def _render_status_panel(self) -> Panel:
        status = Text()
        status.append("Status: ", style="bold")
        status_style = "green" if self.ui_status == "Listening" else "yellow"
        status.append(self.ui_status, style=status_style)
        status.append(" | ")
        status.append(f"Language: {self.language}")
        status.append(" | ")
        status.append(f"VAD: mode {self.vad_mode}, {self.vad_frame_ms}ms")
        status.append(" | ")
        status.append(f"ASR: {self._short_model_name(self.model_path)}")
        if self.analyze:
            status.append(" | ")
            status.append(f"LLM: {self._short_model_name(self.llm_model_name)}")
        return Panel(status, title="Status", padding=(0, 1))

    def _render_transcript_panel(self) -> Panel:
        body = Text()
        for transcript, analysis in self.ui_history[-self.ui_max_history:]:
            body.append("> ", style="bold green")
            body.append(transcript)
            body.append("\n")
            if analysis:
                intent = analysis.get("intent", "")
                entities = analysis.get("entities", "")
                action = analysis.get("action", "")
                if intent:
                    body.append("Intent: ", style="cyan")
                    body.append(intent)
                    body.append("\n")
                if entities and entities.lower() != "none":
                    body.append("Entities: ", style="cyan")
                    body.append(entities)
                    body.append("\n")
                if action:
                    body.append("Action: ", style="cyan")
                    body.append(action)
                    body.append("\n")
            body.append("\n")
        if self.ui_partial and not self.turn_complete:
            body.append("... ", style="dim")
            body.append(self.ui_partial, style="dim")
            body.append("\n")
        if not body.plain:
            body.append("Waiting for speech...", style="dim")
        return Panel(body, title="Transcript", padding=(0, 1))

    def _render_stats_panel(self) -> Panel:
        stats = Table.grid(expand=True, padding=(0, 1))
        stats.add_column(justify="right", style="cyan")
        stats.add_column()
        stats.add_row("VAD", self.ui_vad_state)
        stats.add_row("Buffer", f"{self.ui_buffer_seconds:.1f}s")
        stats.add_row("Queue", str(self.ui_queue_size))
        stats.add_row(
            "ASR",
            f"{self.ui_asr_ms:.0f} ms" if self.ui_asr_ms is not None else "--",
        )
        stats.add_row(
            "Analysis",
            f"{self.ui_analysis_ms:.0f} ms" if self.ui_analysis_ms is not None else "--",
        )
        return Panel(stats, title="Stats", padding=(0, 1))

    def _render_ui(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._render_status_panel(), name="status", size=3),
            Layout(self._render_transcript_panel(), name="transcript", ratio=2),
            Layout(self._render_stats_panel(), name="stats", size=7),
        )
        return layout

    def _update_ui(self, force: bool = False) -> None:
        if not self.live:
            return
        self.live.update(self._render_ui(), refresh=force)

    def _log_info(self, message: str, *args: Any) -> None:
        """Log only when the live UI is disabled to avoid terminal clutter."""
        if not self.live:
            LOGGER.info(message, *args)

    def _append_audio(self, frame: np.ndarray) -> None:
        """Avoid reallocations by writing into a ring buffer."""
        if frame.size == 0:
            return
        n = frame.size
        end = self.buffer_write_pos + n
        if end <= self.max_buffer_samples:
            self.audio_buffer[self.buffer_write_pos: end] = frame
        else:
            first = self.max_buffer_samples - self.buffer_write_pos
            self.audio_buffer[self.buffer_write_pos:] = frame[:first]
            self.audio_buffer[: end % self.max_buffer_samples] = frame[first:]
        self.buffer_write_pos = end % self.max_buffer_samples
        self.buffer_filled = min(self.max_buffer_samples, self.buffer_filled + n)
        self.total_samples_written += n
        self.ui_buffer_seconds = self.buffer_filled / self.sample_rate

    def _get_recent_audio(self, seconds: float) -> np.ndarray:
        """Provide a sliding window for periodic ASR updates."""
        if self.buffer_filled == 0:
            return np.array([], dtype=np.int16)
        num = min(int(seconds * self.sample_rate), self.buffer_filled)
        if num <= 0:
            return np.array([], dtype=np.int16)
        start = (self.buffer_write_pos - num) % self.max_buffer_samples
        end = start + num
        if end <= self.max_buffer_samples:
            return self.audio_buffer[start:end].copy()
        return np.concatenate(
            [
                self.audio_buffer[start:],
                self.audio_buffer[: end % self.max_buffer_samples],
            ]
        )

    def _reset_audio_state(self) -> None:
        """Start a fresh turn so state does not bleed across turns."""
        self.buffer_write_pos = 0
        self.buffer_filled = 0
        self.ui_buffer_seconds = 0.0
        self.vad_residual = np.array([], dtype=np.int16)
        self.vad_speech_detected = False
        self.vad_silence_count = 0

    def _update_vad(self, frame: np.ndarray) -> bool:
        """Detect end-of-turn without running ASR constantly."""
        if frame.size == 0:
            return False
        if self.vad_residual.size == 0:
            self.vad_residual = frame.copy()
        else:
            self.vad_residual = np.concatenate([self.vad_residual, frame])

        turn_complete = False
        while self.vad_residual.size >= self.vad_frame_samples:
            chunk = self.vad_residual[: self.vad_frame_samples]
            self.vad_residual = self.vad_residual[self.vad_frame_samples:]
            is_speech = self.vad.is_speech(chunk.tobytes(), self.sample_rate)
            if is_speech:
                self.ui_vad_state = "speech"
                self.vad_speech_detected = True
                self.vad_silence_count = 0
            elif self.vad_speech_detected:
                self.ui_vad_state = "silence"
                self.vad_silence_count += 1
                if self.vad_silence_count >= self.vad_silence_frames:
                    turn_complete = True
                    self.vad_speech_detected = False
                    self.vad_silence_count = 0
                    self.vad_residual = np.array([], dtype=np.int16)
                    break
            else:
                self.ui_vad_state = "silence"
        return turn_complete

    def _transcribe(self, audio: np.ndarray) -> str:
        """Centralize ASR calls so MLX use stays serialized."""
        if len(audio) < self.sample_rate * 0.3:  # Avoid low-signal ASR calls.
            return ""
        parts = []
        with _suppress_output():
            for token in transcribe(
                    self.model, self.tokenizer, self.feature_extractor, audio, self.language
            ):
                parts.append(token)
        return "".join(parts).strip()

    def _is_meaningful(self, text: str) -> bool:
        """Filter out noise so we do not finalize junk output."""
        # Normalize for a minimal-content check.
        cleaned = re.sub(r"[^\w]", "", text)
        return len(cleaned) >= 2  # Avoid short spurious bursts.

    def _analyze_intent(self, text: str) -> Dict[str, str]:
        """Normalize intent output into stable fields for display."""
        from mlx_lm.generate import generate

        messages = [
            {"role": "user", "content": INTENT_EXPLAIN_PROMPT.format(text=text)}
        ]
        prompt = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with _suppress_output():
            response = generate(
                self.llm, self.llm_tokenizer, prompt, max_tokens=100, verbose=False
            )

        # Keep output clean; some models emit reasoning tags.
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()

        # Normalize into stable fields for display.
        result = {"intent": "", "entities": "", "action": ""}
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                result["intent"] = line[7:].strip()
            elif line.upper().startswith("ENTITIES:"):
                result["entities"] = line[9:].strip()
            elif line.upper().startswith("ACTION:"):
                result["action"] = line[7:].strip()

        return result

    async def _handle_turn_complete(self) -> None:
        """Emit a final turn result once VAD indicates completion."""
        if (
                not self.current_transcript
                or self.current_transcript == self.last_transcript
        ):
            return
        if not self._is_meaningful(self.current_transcript):
            return
        if len(self.current_transcript.split()) < self.min_words:
            return

        self.turn_complete = True
        final_transcript = self.current_transcript

        analysis_result = None
        if self.analyze:
            self.ui_status = "Analyzing"
            self._update_ui()
            async with self.gpu_lock:
                start = time.perf_counter()
                analysis_result = await asyncio.to_thread(
                    self._analyze_intent, final_transcript
                )
                self.ui_analysis_ms = (time.perf_counter() - start) * 1000

        self.pending_analysis = (final_transcript, analysis_result)
        self.last_transcript = final_transcript
        self.current_transcript = ""
        self.ui_status = "Listening"

        async with self.buffer_lock:
            self._reset_audio_state()

        self.last_transcribed_sample = self.total_samples_written
        self.turn_complete = False

    async def _processor(self):
        """Coordinate capture/VAD/ASR in one loop to avoid races."""
        min_new_samples = int(self.sample_rate * 0.2)
        last_transcribe = self.loop.time()

        while True:
            self.ui_queue_size = self.audio_queue.qsize()
            frame = None
            try:
                frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

            if frame is not None:
                async with self.buffer_lock:
                    self._append_audio(frame)
                    turn_complete = self._update_vad(frame)

                if turn_complete:
                    await self._handle_turn_complete()

            now = self.loop.time()
            if now - last_transcribe >= self.transcribe_interval:
                if (
                        self.total_samples_written - self.last_transcribed_sample
                        >= min_new_samples
                ):
                    async with self.buffer_lock:
                        audio_int16 = self._get_recent_audio(self.max_buffer_seconds)

                    if audio_int16.size >= int(self.sample_rate * 0.3):
                        if not self.gpu_lock.locked():
                            async with self.gpu_lock:
                                self.ui_status = "Transcribing"
                                self._update_ui()
                                audio = audio_int16.astype(np.float32) / 32768.0
                                start = time.perf_counter()
                                text = await asyncio.to_thread(self._transcribe, audio)
                                self.ui_asr_ms = (time.perf_counter() - start) * 1000
                            if text and text != self.current_transcript:
                                self.current_transcript = text
                            self.ui_status = "Listening"
                        self.last_transcribed_sample = self.total_samples_written

                last_transcribe = now

    async def _display(self):
        """Keep UI responsive without blocking the ASR pipeline."""
        last_displayed = ""
        is_tty = sys.stdout.isatty()

        while True:
            await asyncio.sleep(0.1)

            if self.pending_analysis:
                transcript, analysis = self.pending_analysis
                self.pending_analysis = None

                self.ui_history.append((transcript, analysis))
                self.ui_partial = ""
                self._update_ui(force=True)

                if not is_tty:
                    # Clean output for piping (works with or without live UI)
                    os.write(1, f"{transcript}\n".encode())
                elif not self.live:
                    sys.stdout.write("\r\033[K")
                    self.console_out.print(f"[bold green]>[/bold green] {transcript}")
                    if analysis:
                        if analysis["intent"] or analysis["action"]:
                            self.console_out.print(
                                f"  [cyan]Intent:[/cyan] {analysis['intent']}"
                            )
                            if (
                                    analysis["entities"]
                                    and analysis["entities"].lower() != "none"
                            ):
                                self.console_out.print(
                                    f"  [cyan]Entities:[/cyan] {analysis['entities']}"
                                )
                            if analysis["action"]:
                                self.console_out.print(
                                    f"  [cyan]Action:[/cyan] {analysis['action']}"
                                )
                        self.console_out.print()

                last_displayed = ""
                continue

            if self.current_transcript and self._is_meaningful(self.current_transcript):
                if self.current_transcript != last_displayed and not self.turn_complete:
                    self.ui_partial = self.current_transcript
                    if not self.live:
                        if is_tty:
                            width = shutil.get_terminal_size(fallback=(80, 20)).columns - 5
                            width = max(width, 10)
                            display_text = self.current_transcript
                            if len(display_text) > width:
                                display_text = "..." + display_text[-(width - 3):]

                            sys.stdout.write(f"\r\033[K  \033[2m{display_text}\033[0m")
                            sys.stdout.flush()
                    last_displayed = self.current_transcript

            if self.live:
                self._update_ui()

    async def run(self):
        """Wire models, stream, and tasks, then manage their lifecycle."""
        self.loop = asyncio.get_running_loop()
        if not self.no_ui:
            self.live = Live(
                self._render_ui(),
                console=self.console_ui,
                refresh_per_second=10,
                transient=False,
            )
            self.live.start()

        self.ui_status = "Loading ASR model..."
        self._update_ui(force=True)
        self._log_info("Loading ASR model...")
        self.model, self.tokenizer, self.feature_extractor = await asyncio.to_thread(
            load_qwen3_asr, self.model_path
        )

        # Prime caches and trigger one-time warnings off-screen.
        def warmup():
            import io

            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                dummy_audio = (
                        np.random.randn(self.sample_rate).astype(np.float32) * 0.01
                )
                list(
                    transcribe(
                        self.model,
                        self.tokenizer,
                        self.feature_extractor,
                        dummy_audio,
                        self.language,
                    )
                )
            finally:
                sys.stderr = old_stderr

        await asyncio.to_thread(warmup)

        if self.analyze:
            self.ui_status = "Loading LLM..."
            self._update_ui(force=True)
            self._log_info("Loading LLM...")
            from mlx_lm.utils import load as load_llm

            self.llm, self.llm_tokenizer = await asyncio.to_thread(
                load_llm, self.llm_model_name
            )

        stream_kwargs = {}
        if self.device is not None:
            stream_kwargs["device"] = self.device
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
            **stream_kwargs,
        )

        # Keep status separate from live transcript output.
        info = f"Language: {self.language} | VAD: mode {self.vad_mode}, {self.vad_frame_ms}ms"
        if self.analyze:
            info += " | Analysis: enabled"
        self._log_info("Ready - %s", info)
        self._log_info("Listening... (Ctrl+C to stop)")
        self.ui_status = "Listening"
        self._update_ui(force=True)

        stream.start()

        tasks = [
            asyncio.create_task(self._processor()),
            asyncio.create_task(self._display()),
        ]

        # Ensure cleanup on Ctrl+C.
        stop_event = asyncio.Event()

        def signal_handler():
            if not stop_event.is_set():
                self._log_info("Stopping...")
                stop_event.set()

        signal_handler_installed = False
        try:
            self.loop.add_signal_handler(signal.SIGINT, signal_handler)
            signal_handler_installed = True
        except NotImplementedError:
            signal_handler_installed = False

        try:
            await stop_event.wait()
        finally:
            if signal_handler_installed:
                try:
                    self.loop.remove_signal_handler(signal.SIGINT)
                except Exception:
                    pass

            for t in tasks:
                t.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
                )
            except asyncio.TimeoutError:
                pass

            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

            if self.live:
                self.live.stop()


# =============================================================================
# Main
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Separate CLI configuration for reuse and testability."""
    parser = argparse.ArgumentParser(
        description="Always-on speech transcription with Qwen3-ASR"
    )
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL, help="ASR model")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language")
    parser.add_argument(
        "--transcribe-interval",
        type=float,
        default=DEFAULT_TRANSCRIBE_INTERVAL,
        help=f"How often to update transcription (default: {DEFAULT_TRANSCRIBE_INTERVAL}s)",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=DEFAULT_VAD_FRAME_MS,
        choices=[10, 20, 30],
        help=f"VAD frame size in ms (10/20/30, default: {DEFAULT_VAD_FRAME_MS})",
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        default=DEFAULT_VAD_MODE,
        help=f"VAD aggressiveness 0-3 (default: {DEFAULT_VAD_MODE})",
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=DEFAULT_VAD_SILENCE_MS,
        help=f"Silence required to finalize a turn (default: {DEFAULT_VAD_SILENCE_MS}ms)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum words to finalize a turn (default: {DEFAULT_MIN_WORDS})",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Enable LLM intent analysis on turn completion",
    )
    parser.add_argument("--llm-model", default=None, help="LLM model for analysis")
    parser.add_argument(
        "--no-ui", action="store_true", help="Disable the Rich live UI"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices"
    )
    parser.add_argument("--device", type=int, default=None, help="Audio input device")
    return parser


def list_audio_devices() -> None:
    """Allow device discovery without running the ASR pipeline."""
    console = Console()
    table = Table(title="Audio Input Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Device", style="white")
    table.add_column("Default", style="green")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            is_default = "Yes" if i == sd.default.device[0] else ""
            table.add_row(str(i), d["name"], is_default)
    console.print(table)


def main() -> int:
    """Provide a CLI entry that returns an exit code."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=False,
                show_path=False,
                rich_tracebacks=False,
            )
        ],
    )
    # Silence chatty HTTP request logs from model downloads.
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    transcriber = RealtimeTranscriber(
        model_path=args.model,
        language=args.language,
        transcribe_interval=args.transcribe_interval,
        vad_frame_ms=args.vad_frame_ms,
        vad_mode=args.vad_mode,
        vad_silence_ms=args.vad_silence_ms,
        min_words=args.min_words,
        analyze=args.analyze,
        llm_model=args.llm_model,
        device=args.device,
        no_ui=args.no_ui,
    )

    asyncio.run(transcriber.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
