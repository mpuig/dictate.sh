# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx>=0.22.0",
#     "mlx-lm>=0.22.0",
#     "numpy",
#     "sounddevice",
#     "transformers",
#     "huggingface_hub",
#     "rich",
#     "onnxruntime",
#     "pyannote.audio",
#     "torch",
# ]
# ///
"""
Standalone, low-latency transcription with smart turn detection for Apple Silicon.

Uses MLX for ASR inference and an ONNX acoustic model (pipecat-ai/smart-turn-v3)
to predict turn completion from audio prosody. This prevents premature turn splits
during natural pauses that simple VAD-based silence thresholds would trigger.

Three-layer turn detection:
    1. VAD (Silero VAD ONNX, 200ms default silence) — fast initial trigger
    2. Smart Turn ONNX model (~8MB, ~12ms CPU) — acoustic turn-completion prediction
    3. LLM single-token tagging (optional, --turn-check) — transcript completeness check

Design notes:
    - Rolling-window ASR trades some stability for low latency.
    - VAD triggers Smart Turn check instead of directly finalizing.
    - Smart Turn runs on CPU via onnxruntime — does NOT need GPU lock.
    - MLX is serialized to avoid concurrency issues on the GPU.
    - Logging goes to stderr so transcripts stay clean on stdout.

Troubleshooting:
    - If turns split too often: increase --vad-silence-ms or --smart-turn-threshold.
    - If nothing transcribes: check mic permissions or run --list-devices.
    - If output feels laggy: reduce --transcribe-interval.
Usage:
    uv run stt_turn_by_turn.py
    uv run stt_turn_by_turn.py --model mlx-community/Qwen3-ASR-1.7B-8bit
    uv run stt_turn_by_turn.py --turn-check   # enable LLM layer 3
    uv run stt_turn_by_turn.py --intent       # extract intent JSON per turn
    uv run stt_turn_by_turn.py --diarization  # enable diarization (if available)
    uv run stt_turn_by_turn.py --smart-turn-threshold 0.6

Models (MLX Qwen3-ASR):
    - mlx-community/Qwen3-ASR-0.6B-4bit: fastest, lowest quality.
    - mlx-community/Qwen3-ASR-0.6B-8bit: good balance (default).
    - mlx-community/Qwen3-ASR-0.6B-bf16: higher quality, more RAM.
    - mlx-community/Qwen3-ASR-1.7B-8bit: higher quality, slower.

LLM models for --llm-model (used by --turn-check/--intent; availability may change):
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


@dataclass(frozen=True)
class ASRDefaults:
    model: str = "mlx-community/Qwen3-ASR-0.6B-8bit"
    llm_model: str = "mlx-community/Qwen3-0.6B-4bit"
    language: str = "English"
    sample_rate: int = 16000
    transcribe_interval: float = 0.5
    min_words: int = 3
    max_buffer_seconds: int = 30
    audio_queue_maxsize: int = 200


@dataclass(frozen=True)
class VADDefaults:
    silence_ms: int = 200
    window: int = 512  # 32ms at 16kHz
    context: int = 64
    threshold: float = 0.5


@dataclass(frozen=True)
class SmartTurnDefaults:
    threshold: float = 0.5
    incomplete_short_timeout: float = 5.0
    incomplete_long_timeout: float = 10.0
    audio_seconds: int = 8
    sample_rate: int = 16000


ASR_DEFAULTS = ASRDefaults()
VAD_DEFAULTS = VADDefaults()
SMART_TURN_DEFAULTS = SmartTurnDefaults()

INT16_SCALE = 1.0 / 32768.0
EMPTY_INT16 = np.array([], dtype=np.int16)

ASR_PROMPT_PREFIX = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
)
ASR_PROMPT_SUFFIX_TEMPLATE = (
    "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage {lang}<asr_text>"
)
GPU_BUSY_COOLDOWN = 0.15

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


def build_arg_parser() -> argparse.ArgumentParser:
    """Public CLI builder wrapper (implementation lives below)."""
    return _build_arg_parser()


def list_audio_devices() -> None:
    """Public device lister wrapper (implementation lives below)."""
    _list_audio_devices()


def main() -> int:
    """Public CLI entrypoint wrapper (implementation lives below)."""
    return _main()


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


@dataclass
class BufferState:
    """Ring buffer bookkeeping for audio capture."""

    max_samples: int
    sample_rate: int
    buffer: np.ndarray
    write_pos: int = 0
    filled: int = 0
    total_written: int = 0


@dataclass
class VADState:
    """State tracking for Silero VAD windowing."""

    frame_samples: int
    silence_frames: int
    buffer: np.ndarray
    buffer_fill: int = 0
    speech_detected: bool = False
    silence_count: int = 0


@dataclass
class UIState:
    """Live UI fields to keep rendering logic cohesive."""

    status: str = "Starting"
    partial: str = ""
    history: List[Tuple[str, Optional[Dict[str, Any]], Optional[str]]] = field(
        default_factory=list
    )
    max_history: int = 50
    vad_state: str = "silence"
    vad_prob: Optional[float] = None
    buffer_seconds: float = 0.0
    queue_size: int = 0
    asr_ms: Optional[float] = None
    smart_turn_prob: Optional[float] = None
    smart_turn_ms: Optional[float] = None
    turn_check_result: Optional[str] = None
    intent_ms: Optional[float] = None
    diarization_ms: Optional[float] = None
    wait_remaining: Optional[float] = None
    dropped_frames: int = 0


def _int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """Normalize int16 PCM to float32 [-1, 1]."""
    return audio_int16.astype(np.float32) * INT16_SCALE


def _replace_audio_embeddings(
        inputs_embeds: mx.array,
        input_ids: mx.array,
        audio_features: mx.array,
        audio_token_id: int,
) -> mx.array:
    """Replace audio placeholder embeddings with encoded audio features.

    Pure MLX — no numpy conversion, no host sync.
    """
    audio_mask = input_ids == audio_token_id  # (B, L) or (L,)

    if input_ids.ndim == 1:
        audio_mask = audio_mask[None, :]
        inputs_embeds = inputs_embeds[None, :, :]

    if audio_features.ndim == 2:
        audio_features = audio_features[None, :, :]

    if audio_features.ndim != 3 or inputs_embeds.ndim != 3:
        return inputs_embeds

    # Cumulative index: maps each audio placeholder to its feature vector.
    cum_idx = mx.cumsum(audio_mask.astype(mx.int32), axis=-1) - 1
    cum_idx = mx.maximum(cum_idx, 0)  # (B, L)

    audio_features = audio_features.astype(inputs_embeds.dtype)
    if audio_features.shape[0] == 1 and audio_mask.shape[0] > 1:
        audio_features = mx.broadcast_to(
            audio_features, (audio_mask.shape[0],) + audio_features.shape[1:]
        )

    # Gather audio features at every position, then select via mask.
    expanded_parts = []
    for b in range(int(audio_mask.shape[0])):
        expanded_parts.append(audio_features[b, cum_idx[b]])  # (L, D)
    audio_expanded = mx.stack(expanded_parts, axis=0)  # (B, L, D)
    mask_3d = audio_mask[:, :, None]  # (B, L, 1)
    return mx.where(mask_3d, audio_expanded, inputs_embeds)


class SmartTurnAnalyzer:
    """Acoustic turn-completion predictor using pipecat-ai/smart-turn-v3.

    Downloads the ONNX model from HuggingFace and runs inference on CPU.
    Analyzes the last 8 seconds of audio to predict whether a speaker has
    finished their turn based on prosody and acoustic features.
    """

    REPO_ID = "pipecat-ai/smart-turn-v3"
    MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
    N_MELS = 80
    N_FRAMES = 800

    def __init__(self, threshold: float = SMART_TURN_DEFAULTS.threshold):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        self.threshold = threshold

        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.MODEL_FILE,
        )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so)

        self.feature_extractor = WhisperFeatureExtractor(
            chunk_length=SMART_TURN_DEFAULTS.audio_seconds
        )

    def _prepare_audio(self, audio_float32: np.ndarray) -> np.ndarray:
        """Truncate to last 8s or zero-pad at the beginning if shorter."""
        max_samples = SMART_TURN_DEFAULTS.audio_seconds * SMART_TURN_DEFAULTS.sample_rate
        if len(audio_float32) > max_samples:
            return audio_float32[-max_samples:]
        elif len(audio_float32) < max_samples:
            padding = max_samples - len(audio_float32)
            return np.pad(audio_float32, (padding, 0), mode="constant", constant_values=0)
        return audio_float32

    def predict(self, audio_float32: np.ndarray) -> Dict[str, Any]:
        """Predict turn completion from audio.

        Returns dict with 'prediction' (0=incomplete, 1=complete) and
        'probability' (float 0-1, sigmoid-activated).
        """
        audio = self._prepare_audio(audio_float32)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SMART_TURN_DEFAULTS.sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=SMART_TURN_DEFAULTS.audio_seconds
                       * SMART_TURN_DEFAULTS.sample_rate,
            truncation=True,
            do_normalize=True,
        )

        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        if input_features.shape != (1, self.N_MELS, self.N_FRAMES):
            input_features = input_features.reshape((1, self.N_MELS, self.N_FRAMES))

        outputs = self.session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0].item())
        prediction = 1 if probability > self.threshold else 0

        return {"prediction": prediction, "probability": probability}


SILERO_VAD_WINDOW = VAD_DEFAULTS.window
SILERO_VAD_CONTEXT = VAD_DEFAULTS.context
SILERO_VAD_THRESHOLD = VAD_DEFAULTS.threshold


class SileroVAD:
    """Silero VAD via ONNX. Stateful — call reset() between turns."""

    REPO_ID = "deepghs/silero-vad-onnx"
    MODEL_FILE = "silero_vad.onnx"

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.MODEL_FILE,
        )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so)

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(SILERO_VAD_CONTEXT, dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def reset(self):
        self._state[:] = 0.0
        self._context[:] = 0.0

    def __call__(self, audio_chunk_f32: np.ndarray) -> float:
        inp = np.concatenate([self._context, audio_chunk_f32])[np.newaxis, :]
        outs = self.session.run(
            None,
            {"input": inp, "state": self._state, "sr": self._sr},
        )
        prob = float(outs[0].item())
        self._state = outs[1]
        self._context = audio_chunk_f32[-SILERO_VAD_CONTEXT:]
        return prob


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
            inputs_embeds = _replace_audio_embeddings(
                inputs_embeds, input_ids, audio_features, self.config.audio_token_id
            )

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


_REPETITION_THRESHOLD = 20


def _detect_repetition(tokens: list[int]) -> bool:
    """Detect degenerate token loops (single-token runs or short patterns)."""
    if len(tokens) < _REPETITION_THRESHOLD:
        return False
    # Single token repeated consecutively.
    count = 1
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] == tokens[i - 1]:
            count += 1
            if count >= _REPETITION_THRESHOLD:
                return True
        else:
            break
    # Pattern of 2-10 tokens repeating.
    for plen in range(2, min(11, len(tokens) // 2 + 1)):
        pattern = tokens[-plen:]
        reps = 1
        pos = len(tokens) - plen * 2
        while pos >= 0:
            if tokens[pos:pos + plen] == pattern:
                reps += 1
                pos -= plen
            else:
                break
        if reps >= max(2, _REPETITION_THRESHOLD // plen):
            return True
    return False


def transcribe(
        model: Qwen3ASRModel,
        tokenizer: TokenizerLike,
        feature_extractor: FeatureExtractorLike,
        audio: np.ndarray,
        language: str = ASR_DEFAULTS.language,
        max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """Stream tokens to keep transcription latency low."""
    from mlx_lm.generate import generate_step

    # Match the model's expected feature pipeline.
    audio_inputs = feature_extractor(
        audio,
        sampling_rate=ASR_DEFAULTS.sample_rate,
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

    audio_pad = "<|audio_pad|>" * num_audio_tokens
    prompt = (
        f"{ASR_PROMPT_PREFIX}{audio_pad}"
        f"{ASR_PROMPT_SUFFIX_TEMPLATE.format(lang=lang_name)}"
    )
    input_ids = mx.array(tokenizer.encode(prompt, return_tensors="np"))

    audio_features = model.get_audio_features(input_features, feature_attention_mask)
    inputs_embeds = model.model.embed_tokens(input_ids)
    inputs_embeds = _replace_audio_embeddings(
        inputs_embeds, input_ids, audio_features, model.config.audio_token_id
    )
    mx.eval(inputs_embeds)
    input_embeddings = inputs_embeds[0]
    prompt_ids = input_ids[0] if input_ids.ndim > 1 else input_ids

    eos_token_ids = [151645, 151643]
    recent_tokens: list[int] = []

    for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=input_embeddings,
            model=model,
            max_tokens=max_tokens,
    ):
        if token in eos_token_ids:
            break
        recent_tokens.append(token)
        # Stop if a single token repeats 20+ times or a 2-10 token pattern
        # loops excessively — prevents infinite generation on stuck patterns.
        if _detect_repetition(recent_tokens):
            break
        yield tokenizer.decode([int(token)])


TURN_CHECK_PROMPT = """You are a speech turn-completion classifier. Given a transcript, respond with ONLY one of these markers:
\u2713 — the utterance is a complete thought or sentence
\u25cb — short and clearly incomplete (fragment, trailing conjunction, etc.)
\u25d0 — long but trailing off or mid-thought

Transcript: "{text}" /no_think"""

INTENT_SCHEMA_JSON = """{
  "$id": "https://example.com/intent-classification.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "IntentClassification",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "intent": {
      "type": "string",
      "enum": [
        "inform",
        "question",
        "evaluate",
        "request_action",
        "complaint",
        "other"
      ],
      "description": "Primary communicative intent of the user input."
    },
    "description": {
      "type": "string",
      "maxLength": 120,
      "description": "One short sentence starting with a verb that describes the user's communicative action."
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Model confidence score between 0 and 1."
    }
  },
  "required": [
    "intent",
    "description"
  ]
}"""

INTENT_PROMPT = """You are an intent classifier. Return ONLY a JSON object that matches the schema.
Use double quotes, no trailing commas, and no extra keys.
Ensure description starts with a verb and is <= 120 characters.

Schema:
{schema}

Transcript: "{text}"
/no_think"""


class RealtimeTranscriber:
    """Encapsulates the async pipeline so capture, VAD, Smart Turn, and ASR stay coordinated."""

    def __init__(
            self,
            model_path: str = ASR_DEFAULTS.model,
            language: str = ASR_DEFAULTS.language,
            transcribe_interval: float = ASR_DEFAULTS.transcribe_interval,
            vad_silence_ms: int = VAD_DEFAULTS.silence_ms,
            min_words: int = ASR_DEFAULTS.min_words,
            turn_check: bool = False,
            intent: bool = False,
            diarization: bool = False,
            llm_model: Optional[str] = None,
            device: Optional[int] = None,
            no_ui: bool = False,
            smart_turn_threshold: float = SMART_TURN_DEFAULTS.threshold,
            incomplete_short_timeout: float = SMART_TURN_DEFAULTS.incomplete_short_timeout,
            incomplete_long_timeout: float = SMART_TURN_DEFAULTS.incomplete_long_timeout,
    ):
        self.model_path = model_path
        self.language = language
        self.transcribe_interval = transcribe_interval
        self.vad_silence_ms = vad_silence_ms
        self.min_words = min_words
        self.turn_check = turn_check
        self.intent = intent
        self.diarization = diarization
        self.llm_model_name = llm_model or ASR_DEFAULTS.llm_model
        self.device = device
        self.no_ui = no_ui
        self.smart_turn_threshold = smart_turn_threshold
        self.incomplete_short_timeout = incomplete_short_timeout
        self.incomplete_long_timeout = incomplete_long_timeout

        self.sample_rate = ASR_DEFAULTS.sample_rate
        # Bound RAM and latency by limiting the rolling window.
        self.max_buffer_seconds = ASR_DEFAULTS.max_buffer_seconds

        self.audio_queue = asyncio.Queue(maxsize=ASR_DEFAULTS.audio_queue_maxsize)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.model = None
        self.tokenizer: Optional[TokenizerLike] = None
        self.feature_extractor: Optional[FeatureExtractorLike] = None
        self.llm = None
        self.llm_tokenizer: Optional[TokenizerLike] = None
        self.smart_turn: Optional[SmartTurnAnalyzer] = None
        self.diarizer = None
        self.diarization_label_map: Dict[str, str] = {}

        # Ring buffer avoids reallocations; int16 matches the input stream.
        self.max_buffer_samples = int(self.max_buffer_seconds * self.sample_rate)
        self.buffer = BufferState(
            max_samples=self.max_buffer_samples,
            sample_rate=self.sample_rate,
            buffer=np.zeros(self.max_buffer_samples, dtype=np.int16),
        )
        self.last_transcribed_sample = 0
        self.last_turn_sample = 0
        self.buffer_lock = asyncio.Lock()

        # MLX is not re-entrant; serialize GPU work to avoid races.
        self.gpu_lock = asyncio.Lock()
        self.gpu_cooldown_until = 0.0

        self.silero_vad: Optional[SileroVAD] = None  # loaded in run()
        self.vad_frame_samples = SILERO_VAD_WINDOW
        vad_silence_frames = int(math.ceil(self.vad_silence_ms / 32))
        self.vad_state = VADState(
            frame_samples=self.vad_frame_samples,
            silence_frames=vad_silence_frames,
            buffer=np.empty(self.vad_frame_samples * 4, dtype=np.int16),
        )
        self.frame_size = self.vad_frame_samples

        # Track output across updates.
        self.current_transcript = ""
        self.last_transcript = ""
        self.turn_complete = False
        self.pending_turn: Optional[
            Tuple[str, Optional[Dict[str, Any]], Optional[str]]
        ] = None

        # Wait-state for LLM turn-check incomplete results.
        self._wait_cancel = asyncio.Event()
        self._in_wait_state = False

        # Rich UI state (stderr) + clean transcript output (stdout).
        self.console_out = Console()
        self.console_ui = Console(stderr=True, force_terminal=True, force_interactive=True)
        self.live: Optional[Live] = None
        self.ui = UIState()

    def _audio_callback(self, indata, frames, time_info, status):
        """Keep callback lightweight by deferring work to the async loop."""
        data = indata.reshape(-1).copy()

        def enqueue():
            if self.audio_queue.full():
                self.ui.dropped_frames += 1
                return
            self.audio_queue.put_nowait(data)

        self.loop.call_soon_threadsafe(enqueue)

    def _short_model_name(self, name: Optional[str]) -> str:
        """Render a short model name for UI labels."""
        if not name:
            return "--"
        return name.split("/")[-1]

    def _render_status_panel(self) -> Panel:
        status = Text()
        status.append("Status: ", style="bold")
        status_style = "green" if self.ui.status == "Listening" else "yellow"
        status.append(self.ui.status, style=status_style)
        status.append(" | ")
        status.append(f"Language: {self.language}")
        status.append(" | ")
        status.append("VAD: Silero")
        status.append(" | ")
        status.append(f"ASR: {self._short_model_name(self.model_path)}")
        if self.turn_check or self.intent:
            status.append(" | ")
            status.append(f"LLM: {self._short_model_name(self.llm_model_name)}")
        if self.diarization:
            status.append(" | ")
            status.append("Diarization: on")
        return Panel(status, title="Status", padding=(0, 1))

    def _render_transcript_panel(self) -> Panel:
        lines: List[Tuple[str, Optional[str]]] = []

        def add_line(text: str, style: Optional[str] = None) -> None:
            lines.append((text, style))

        for transcript, intent, speaker in self.ui.history[-self.ui.max_history:]:
            add_line(f"> {transcript}", "bold green")
            if speaker:
                add_line(f"Speaker: {speaker}", "magenta")
            if intent:
                intent_name = intent.get("intent", "")
                intent_desc = intent.get("description", "")
                intent_conf = intent.get("confidence", None)
                if intent_name:
                    add_line(f"Intent: {intent_name}", "cyan")
                if intent_desc:
                    add_line(f"Description: {intent_desc}", "cyan")
                if intent_conf is not None:
                    add_line(f"Confidence: {intent_conf:.2f}", "cyan")
            add_line("")

        if self.ui.partial and not self.turn_complete:
            add_line(f"... {self.ui.partial}", "dim")

        if not lines:
            add_line("Waiting for speech...", "dim")

        term_lines = shutil.get_terminal_size(fallback=(80, 24)).lines
        reserved = 3 + 9 + 4  # status + stats + panel borders
        max_lines = max(5, term_lines - reserved)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]

        body = Text()
        for text, style in lines:
            body.append(text, style=style)
            body.append("\n")
        return Panel(body, title="Transcript", padding=(0, 1))

    def _render_stats_panel(self) -> Panel:
        stats = Table.grid(expand=True, padding=(0, 1))
        stats.add_column(justify="right", style="cyan")
        stats.add_column()
        vad_text = self.ui.vad_state
        if self.ui.vad_prob is not None:
            vad_text = f"{self.ui.vad_state} ({self.ui.vad_prob:.0%})"
        stats.add_row("VAD", vad_text)
        stats.add_row("Buffer", f"{self.ui.buffer_seconds:.1f}s")
        stats.add_row("Queue", str(self.ui.queue_size))
        stats.add_row("Dropped", str(self.ui.dropped_frames))
        stats.add_row(
            "ASR",
            f"{self.ui.asr_ms:.0f} ms" if self.ui.asr_ms is not None else "--",
        )
        # Smart Turn stats.
        st_text = "--"
        if self.ui.smart_turn_prob is not None:
            prob_pct = self.ui.smart_turn_prob * 100
            ms_str = (
                f"{self.ui.smart_turn_ms:.0f} ms"
                if self.ui.smart_turn_ms is not None
                else ""
            )
            st_text = f"{prob_pct:.0f}% ({ms_str})"
        stats.add_row("Smart Turn", st_text)
        # Turn check result.
        if self.turn_check:
            tc_text = self.ui.turn_check_result or "--"
            stats.add_row("Turn Check", tc_text)
        if self.intent:
            stats.add_row(
                "Intent",
                f"{self.ui.intent_ms:.0f} ms" if self.ui.intent_ms is not None else "--",
            )
        if self.diarization:
            stats.add_row(
                "Diarization",
                f"{self.ui.diarization_ms:.0f} ms"
                if self.ui.diarization_ms is not None
                else "--",
            )
        # Wait state countdown.
        if self.ui.wait_remaining is not None:
            stats.add_row("Wait", f"{self.ui.wait_remaining:.1f}s")
        return Panel(stats, title="Stats", padding=(0, 1))

    def _render_ui(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._render_status_panel(), name="status", size=3),
            Layout(self._render_transcript_panel(), name="transcript", ratio=2),
            Layout(self._render_stats_panel(), name="stats", size=9),
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
        buf = self.buffer
        end = buf.write_pos + n
        if end <= buf.max_samples:
            buf.buffer[buf.write_pos: end] = frame
        else:
            first = buf.max_samples - buf.write_pos
            buf.buffer[buf.write_pos:] = frame[:first]
            buf.buffer[: end % buf.max_samples] = frame[first:]
        buf.write_pos = end % buf.max_samples
        buf.filled = min(buf.max_samples, buf.filled + n)
        buf.total_written += n
        self.ui.buffer_seconds = buf.filled / buf.sample_rate

    def _get_recent_audio(self, seconds: float) -> np.ndarray:
        """Provide a sliding window for periodic ASR updates."""
        buf = self.buffer
        if buf.filled == 0:
            return EMPTY_INT16
        num = min(int(seconds * buf.sample_rate), buf.filled)
        if num <= 0:
            return EMPTY_INT16
        start = (buf.write_pos - num) % buf.max_samples
        end = start + num
        out = np.empty(num, dtype=np.int16)
        if end <= buf.max_samples:
            out[:] = buf.buffer[start:end]
            return out
        first = buf.max_samples - start
        out[:first] = buf.buffer[start:]
        out[first:] = buf.buffer[: end % buf.max_samples]
        return out

    def _get_audio_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Return audio between global sample indices [start_sample, end_sample)."""
        buf = self.buffer
        if buf.filled == 0:
            return EMPTY_INT16

        end_sample = min(end_sample, buf.total_written)
        earliest = buf.total_written - buf.filled
        start_sample = max(start_sample, earliest)
        if end_sample <= start_sample:
            return EMPTY_INT16

        num = end_sample - start_sample
        start_offset = (buf.write_pos - buf.filled) % buf.max_samples
        rel = start_sample - earliest
        start_idx = (start_offset + rel) % buf.max_samples

        out = np.empty(num, dtype=np.int16)
        end_idx = start_idx + num
        if end_idx <= buf.max_samples:
            out[:] = buf.buffer[start_idx:end_idx]
            return out
        first = buf.max_samples - start_idx
        out[:first] = buf.buffer[start_idx:]
        out[first:] = buf.buffer[: end_idx % buf.max_samples]
        return out

    def _reset_audio_state(self) -> None:
        """Start a fresh turn so state does not bleed across turns."""
        self.buffer.write_pos = 0
        self.buffer.filled = 0
        self.ui.buffer_seconds = 0.0
        self.vad_state.buffer_fill = 0
        self.vad_state.speech_detected = False
        self.vad_state.silence_count = 0
        if self.silero_vad is not None:
            self.silero_vad.reset()

    def _update_vad(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        state = self.vad_state
        if state.buffer_fill + frame.size > state.buffer.size:
            new_size = max(state.buffer.size * 2, state.buffer_fill + frame.size)
            new_buf = np.empty(new_size, dtype=np.int16)
            if state.buffer_fill:
                new_buf[: state.buffer_fill] = state.buffer[: state.buffer_fill]
            state.buffer = new_buf
        state.buffer[state.buffer_fill: state.buffer_fill + frame.size] = frame
        state.buffer_fill += frame.size

        turn_complete = False
        while state.buffer_fill >= state.frame_samples:
            chunk_i16 = state.buffer[: state.frame_samples].copy()
            remaining = state.buffer_fill - state.frame_samples
            if remaining:
                state.buffer[:remaining] = state.buffer[
                    state.frame_samples: state.frame_samples + remaining
                ]
            state.buffer_fill = remaining
            chunk_f32 = _int16_to_float32(chunk_i16)
            prob = self.silero_vad(chunk_f32)
            self.ui.vad_prob = prob
            is_speech = prob >= SILERO_VAD_THRESHOLD
            if is_speech:
                self.ui.vad_state = "speech"
                state.speech_detected = True
                state.silence_count = 0
                if self._in_wait_state:
                    self._wait_cancel.set()
            elif state.speech_detected:
                self.ui.vad_state = "silence"
                state.silence_count += 1
                if state.silence_count >= state.silence_frames:
                    turn_complete = True
                    state.speech_detected = False
                    state.silence_count = 0
                    state.buffer_fill = 0
                    break
            else:
                self.ui.vad_state = "silence"
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

    def _check_turn_completion_llm(self, text: str) -> str:
        """Classify transcript completeness via LLM. Returns one of: \u2713, \u25cb, \u25d0."""
        from mlx_lm.generate import generate

        messages = [
            {"role": "user", "content": TURN_CHECK_PROMPT.format(text=text)}
        ]
        prompt = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with _suppress_output():
            response = generate(
                self.llm, self.llm_tokenizer, prompt, max_tokens=5, verbose=False
            )

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()

        if response:
            first_char = response[0]
            if first_char in ("\u2713", "\u25cb", "\u25d0"):
                return first_char

        # Fallback: treat as complete if we can't parse the marker.
        return "\u2713"

    def _parse_intent_payload(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        allowed = {"intent", "description", "confidence"}
        if any(key not in allowed for key in payload.keys()):
            return None

        intent = payload.get("intent")
        description = payload.get("description")
        if not isinstance(intent, str) or not isinstance(description, str):
            return None

        intent = intent.strip()
        description = description.strip()
        if intent not in {
            "inform",
            "question",
            "evaluate",
            "request_action",
            "complaint",
            "other",
        }:
            return None
        if not description:
            return None
        if len(description) > 120:
            description = description[:120].rstrip()

        confidence = payload.get("confidence", None)
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and not (0.0 <= confidence <= 1.0):
                confidence = None

        result = {"intent": intent, "description": description}
        if confidence is not None:
            result["confidence"] = confidence
        return result

    def _classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent using structured JSON output."""
        from mlx_lm.generate import generate

        messages = [
            {
                "role": "user",
                "content": INTENT_PROMPT.format(schema=INTENT_SCHEMA_JSON, text=text),
            }
        ]
        prompt = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with _suppress_output():
            response = generate(
                self.llm, self.llm_tokenizer, prompt, max_tokens=200, verbose=False
            )

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()
        parsed = self._parse_intent_payload(response)
        if parsed is not None:
            return parsed

        return {"intent": "other", "description": "Respond to the user.", "confidence": 0.0}

    def _load_diarizer(self):
        """Load pyannote diarization pipeline if enabled."""
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "Diarization requires 'pyannote.audio'. Install with: "
                "pip install pyannote.audio"
            ) from exc

        token = (
                os.environ.get("PYANNOTE_AUTH_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HF_TOKEN")
                or ""
        )
        model_id = os.environ.get(
            "PYANNOTE_MODEL_ID", "pyannote/speaker-diarization-3.1"
        )
        kwargs = {}
        if token:
            # pyannote has used both `use_auth_token` and `token` across versions.
            try:
                self.diarizer = Pipeline.from_pretrained(
                    model_id, use_auth_token=token
                )
                return
            except TypeError:
                kwargs["token"] = token
        self.diarizer = Pipeline.from_pretrained(model_id, **kwargs)

    def _normalize_speaker_label(self, label: str) -> str:
        if label not in self.diarization_label_map:
            self.diarization_label_map[label] = (
                f"SPEAKER_{len(self.diarization_label_map):02d}"
            )
        return self.diarization_label_map[label]

    def _infer_speaker(self, audio_int16: np.ndarray) -> Optional[str]:
        """Run diarization and return dominant speaker label."""
        if audio_int16.size == 0 or self.diarizer is None:
            return None
        audio_float = _int16_to_float32(audio_int16)
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Diarization requires PyTorch (torch). Install with: "
                "pip install torch"
            ) from exc

        waveform = torch.from_numpy(audio_float).unsqueeze(0)
        diarization = self.diarizer(
            {"waveform": waveform, "sample_rate": self.sample_rate}
        )
        durations: Dict[str, float] = {}
        if hasattr(diarization, "itertracks"):
            for segment, _, label in diarization.itertracks(yield_label=True):
                start = float(getattr(segment, "start", 0.0))
                end = float(getattr(segment, "end", start))
                if end <= start:
                    continue
                norm_label = self._normalize_speaker_label(str(label))
                durations[norm_label] = durations.get(norm_label, 0.0) + (end - start)
        if not durations:
            return None
        return max(durations.items(), key=lambda kv: kv[1])[0]

    async def _enter_wait_state(self, timeout: float) -> bool:
        """Wait for timeout, cancellable by speech resumption.

        Returns True if the wait completed (should finalize), False if cancelled.
        """
        self._wait_cancel.clear()
        self._in_wait_state = True
        self.ui.status = f"Waiting ({timeout:.0f}s)"
        self._update_ui()

        start = time.monotonic()
        try:
            # Poll in small intervals so we can update the countdown.
            while True:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                self.ui.wait_remaining = max(0.0, remaining)
                self._update_ui()

                if remaining <= 0:
                    return True

                try:
                    await asyncio.wait_for(
                        self._wait_cancel.wait(),
                        timeout=min(0.2, remaining),
                    )
                    # Event was set — speech resumed, cancel the wait.
                    return False
                except asyncio.TimeoutError:
                    continue
        finally:
            self._in_wait_state = False
            self.ui.wait_remaining = None
            self._wait_cancel.clear()

    async def _finalize_turn(self, final_transcript: str) -> None:
        """Emit a final turn result and reset state for the next turn."""
        intent_result = None
        speaker_label = None

        diarization_audio = EMPTY_INT16
        async with self.buffer_lock:
            end_sample = self.buffer.total_written
            if self.diarization:
                diarization_audio = self._get_audio_window(
                    self.last_turn_sample, end_sample
                )
            self.last_turn_sample = end_sample

        if self.diarization:
            self.ui.status = "Diarizing"
            self._update_ui()
            start = time.perf_counter()
            speaker_label = await asyncio.to_thread(
                self._infer_speaker, diarization_audio
            )
            self.ui.diarization_ms = (time.perf_counter() - start) * 1000

        if self.intent:
            self.ui.status = "Intent"
            self._update_ui()
            async with self.gpu_lock:
                start = time.perf_counter()
                intent_result = await asyncio.to_thread(
                    self._classify_intent, final_transcript
                )
                self.ui.intent_ms = (time.perf_counter() - start) * 1000

        self.pending_turn = (final_transcript, intent_result, speaker_label)
        self.last_transcript = final_transcript
        self.current_transcript = ""
        self.ui.status = "Listening"

        async with self.buffer_lock:
            self._reset_audio_state()

        self.last_transcribed_sample = self.buffer.total_written
        self.turn_complete = False

    async def _refresh_transcript_for_turn(self) -> None:
        """Run a quick transcription to support turn-completion checks."""
        async with self.buffer_lock:
            audio_int16 = self._get_recent_audio(self.max_buffer_seconds)

        if audio_int16.size < int(self.sample_rate * 0.3):
            return
        if self.gpu_lock.locked():
            return
        async with self.gpu_lock:
            audio_float = _int16_to_float32(audio_int16)
            start = time.perf_counter()
            text = await asyncio.to_thread(self._transcribe, audio_float)
            self.ui.asr_ms = (time.perf_counter() - start) * 1000
        if text:
            self.current_transcript = text

    def _is_turn_candidate(self) -> bool:
        """Check whether the current transcript is valid to finalize."""
        if not self.current_transcript or self.current_transcript == self.last_transcript:
            return False
        if not self._is_meaningful(self.current_transcript):
            return False
        if len(self.current_transcript.split()) < self.min_words:
            return False
        return True

    async def _smart_turn_allows_finalize(self) -> bool:
        """Run Smart Turn to decide if the speaker has completed a turn."""
        async with self.buffer_lock:
            smart_turn_audio = self._get_recent_audio(
                SMART_TURN_DEFAULTS.audio_seconds
            )
        smart_turn_float = _int16_to_float32(smart_turn_audio)

        start = time.perf_counter()
        result = await asyncio.to_thread(self.smart_turn.predict, smart_turn_float)
        self.ui.smart_turn_ms = (time.perf_counter() - start) * 1000
        self.ui.smart_turn_prob = result["probability"]
        self._update_ui()

        if result["prediction"] == 0:
            self._log_info(
                "Smart Turn: incomplete (prob=%.2f), continuing...",
                result["probability"],
            )
            self.vad_state.speech_detected = True
            self.vad_state.silence_count = 0
            return False

        self._log_info(
            "Smart Turn: complete (prob=%.2f)",
            result["probability"],
        )
        return True

    async def _handle_turn_check(self, final_transcript: str) -> bool:
        """Apply optional LLM turn-checks. Returns True if we should finalize."""
        if not self.turn_check:
            return True

        self.ui.status = "Turn Check"
        self._update_ui()

        async with self.gpu_lock:
            marker = await asyncio.to_thread(
                self._check_turn_completion_llm, final_transcript
            )
        self.ui.turn_check_result = marker
        self._update_ui()

        if marker == "\u2713":
            return True
        if marker == "\u25cb":
            timeout = self.incomplete_short_timeout
        else:
            timeout = self.incomplete_long_timeout

        self._log_info("Turn check: %s, waiting %.0fs...", marker, timeout)
        should_finalize = await self._enter_wait_state(timeout)

        if should_finalize:
            return True

        self._log_info("Wait cancelled — speech resumed")
        self.turn_complete = False
        self.ui.status = "Listening"
        self._update_ui()
        return False

    async def _handle_turn_complete(self) -> None:
        """Process a VAD silence trigger through the smart turn pipeline.

        Flow: VAD trigger -> Smart Turn ONNX check -> [LLM turn check] -> finalize
        """
        await self._refresh_transcript_for_turn()

        if not self._is_turn_candidate():
            return

        if not await self._smart_turn_allows_finalize():
            return

        self.turn_complete = True
        final_transcript = self.current_transcript

        if await self._handle_turn_check(final_transcript):
            await self._finalize_turn(final_transcript)

    async def _processor(self):
        """Coordinate capture/VAD/ASR in one loop to avoid races."""
        min_new_samples = int(self.sample_rate * 0.2)
        last_transcribe = self.loop.time()

        while True:
            self.ui.queue_size = self.audio_queue.qsize()
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
                        self.buffer.total_written - self.last_transcribed_sample
                        >= min_new_samples
                ):
                    async with self.buffer_lock:
                        audio_int16 = self._get_recent_audio(self.max_buffer_seconds)

                    if audio_int16.size >= int(self.sample_rate * 0.3):
                        if self.gpu_lock.locked():
                            self.gpu_cooldown_until = now + GPU_BUSY_COOLDOWN
                        elif now >= self.gpu_cooldown_until:
                            async with self.gpu_lock:
                                self.ui.status = "Transcribing"
                                self._update_ui()
                                audio = _int16_to_float32(audio_int16)
                                start = time.perf_counter()
                                text = await asyncio.to_thread(self._transcribe, audio)
                                self.ui.asr_ms = (time.perf_counter() - start) * 1000
                            if text and text != self.current_transcript:
                                self.current_transcript = text
                            self.ui.status = "Listening"
                        self.last_transcribed_sample = self.buffer.total_written

                last_transcribe = now

    async def _display(self):
        """Keep UI responsive without blocking the ASR pipeline."""
        last_displayed = ""
        is_tty = sys.stdout.isatty()

        while True:
            await asyncio.sleep(0.1)

            if self.pending_turn:
                transcript, intent, speaker = self.pending_turn
                self.pending_turn = None

                self.ui.history.append((transcript, intent, speaker))
                self.ui.partial = ""
                self._update_ui(force=True)

                if not is_tty:
                    # Clean output for piping (works with or without live UI)
                    os.write(1, f"{transcript}\n".encode())
                elif not self.live:
                    sys.stdout.write("\r\033[K")
                    self.console_out.print(f"[bold green]>[/bold green] {transcript}")
                    if speaker:
                        self.console_out.print(
                            f"  [magenta]Speaker:[/magenta] {speaker}"
                        )
                    if intent:
                        self.console_out.print(
                            f"  [cyan]Intent:[/cyan] {intent.get('intent', '')}"
                        )
                        self.console_out.print(
                            f"  [cyan]Description:[/cyan] {intent.get('description', '')}"
                        )
                        conf = intent.get("confidence", None)
                        if isinstance(conf, (int, float)):
                            self.console_out.print(
                                f"  [cyan]Confidence:[/cyan] {conf:.2f}"
                            )
                        self.console_out.print()

                last_displayed = ""
                continue

            if self.current_transcript and self._is_meaningful(self.current_transcript):
                if self.current_transcript != last_displayed and not self.turn_complete:
                    self.ui.partial = self.current_transcript
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

        self.ui.status = "Loading ASR model..."
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

        # Both run on CPU via onnxruntime — no GPU lock needed.
        self.ui.status = "Loading VAD + Smart Turn models..."
        self._update_ui(force=True)
        self._log_info("Loading Silero VAD model...")
        self.silero_vad = await asyncio.to_thread(SileroVAD)
        self._log_info("Loading Smart Turn model...")
        self.smart_turn = await asyncio.to_thread(
            SmartTurnAnalyzer, self.smart_turn_threshold
        )

        if self.diarization:
            self.ui.status = "Loading Diarization..."
            self._update_ui(force=True)
            self._log_info("Loading diarization pipeline...")
            await asyncio.to_thread(self._load_diarizer)

        if self.turn_check or self.intent:
            self.ui.status = "Loading LLM..."
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
        info = f"Language: {self.language} | VAD: Silero, {self.vad_silence_ms}ms silence"
        info += f" | Smart Turn: threshold {self.smart_turn_threshold}"
        if self.turn_check:
            info += " | Turn Check: enabled"
        if self.intent:
            info += " | Intent: enabled"
        if self.diarization:
            info += " | Diarization: enabled"
        self._log_info("Ready - %s", info)
        self._log_info("Listening... (Ctrl+C to stop)")
        self.ui.status = "Listening"
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Separate CLI configuration for reuse and testability."""
    parser = argparse.ArgumentParser(
        description="Speech transcription with smart turn detection (VAD + ONNX + optional LLM)"
    )
    parser.add_argument("--model", default=ASR_DEFAULTS.model, help="ASR model")
    parser.add_argument("--language", default=ASR_DEFAULTS.language, help="Language")
    parser.add_argument(
        "--transcribe-interval",
        type=float,
        default=ASR_DEFAULTS.transcribe_interval,
        help=(
            "How often to update transcription "
            f"(default: {ASR_DEFAULTS.transcribe_interval}s)"
        ),
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=VAD_DEFAULTS.silence_ms,
        help=(
            "Silence to trigger smart turn check "
            f"(default: {VAD_DEFAULTS.silence_ms}ms)"
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=ASR_DEFAULTS.min_words,
        help=(
            "Minimum words to finalize a turn "
            f"(default: {ASR_DEFAULTS.min_words})"
        ),
    )
    parser.add_argument(
        "--smart-turn-threshold",
        type=float,
        default=SMART_TURN_DEFAULTS.threshold,
        help=(
            "Smart Turn probability threshold "
            f"(default: {SMART_TURN_DEFAULTS.threshold})"
        ),
    )
    parser.add_argument(
        "--turn-check",
        action="store_true",
        help="Enable LLM turn-completion check (layer 3)",
    )
    parser.add_argument(
        "--intent",
        action="store_true",
        help="Enable LLM intent classification per finalized turn",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable diarization (requires additional setup if available)",
    )
    parser.add_argument(
        "--incomplete-short-timeout",
        type=float,
        default=SMART_TURN_DEFAULTS.incomplete_short_timeout,
        help=(
            "Wait time for short-incomplete turns "
            f"(default: {SMART_TURN_DEFAULTS.incomplete_short_timeout}s)"
        ),
    )
    parser.add_argument(
        "--incomplete-long-timeout",
        type=float,
        default=SMART_TURN_DEFAULTS.incomplete_long_timeout,
        help=(
            "Wait time for long-incomplete turns "
            f"(default: {SMART_TURN_DEFAULTS.incomplete_long_timeout}s)"
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model for --turn-check/--intent",
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="Disable the Rich live UI"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices"
    )
    parser.add_argument("--device", type=int, default=None, help="Audio input device")
    return parser


def _list_audio_devices() -> None:
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


def _main() -> int:
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
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        _list_audio_devices()
        return 0

    transcriber = RealtimeTranscriber(
        model_path=args.model,
        language=args.language,
        transcribe_interval=args.transcribe_interval,
        vad_silence_ms=args.vad_silence_ms,
        min_words=args.min_words,
        turn_check=args.turn_check,
        intent=args.intent,
        diarization=args.diarization,
        llm_model=args.llm_model,
        device=args.device,
        no_ui=args.no_ui,
        smart_turn_threshold=args.smart_turn_threshold,
        incomplete_short_timeout=args.incomplete_short_timeout,
        incomplete_long_timeout=args.incomplete_long_timeout,
    )

    asyncio.run(transcriber.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
