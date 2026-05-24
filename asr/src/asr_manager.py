"""Manages the ASR model."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def _load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Load WAV bytes, convert to mono 16 kHz float32."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return np.zeros(target_sr, dtype=np.float32)

    audio = np.nan_to_num(audio)

    # Normalise only if audio is outside normal range.
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32)


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _postprocess(text: str) -> str:
    """Match challenge scoring transforms as much as possible without damaging words."""
    text = text.strip()

    cjk_count = _count_cjk(text)

    # Chinese CER removes whitespace, so remove whitespace if output is Chinese.
    if cjk_count >= 2:
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"\s+", " ", text).strip()

    return text


class ASRManager:
    def __init__(self):
        torch.set_num_threads(int(os.getenv("ASR_CPU_THREADS", "8")))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        src_dir = Path(__file__).resolve().parent

        default_model_dir = src_dir / "asr_runs" / "whisper-medium-rag-ft" / "final"

        self.model_dir = Path(
            os.getenv("ASR_MODEL_DIR", str(default_model_dir))
        )

        # Mixed beam:
        # - beam 1 was better for Chinese in your validation.
        # - beam 3 was slightly better for English/Malay/Tamil.
        self.use_mixed_beam = os.getenv("ASR_MIXED_BEAM", "0") == "1"
        self.num_beams = int(os.getenv("ASR_BEAMS", "1"))

        self.processor = WhisperProcessor.from_pretrained(str(self.model_dir))

        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

        # Important for Advanced track:
        # Do not force English. Let Whisper transcribe multilingual audio.
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.task = "transcribe"

    def _decode(self, input_features: torch.Tensor, beams: int) -> str:
        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                task="transcribe",
                num_beams=beams,
                do_sample=False,
                max_new_tokens=192,
                return_timestamps=False,
            )

        text = self.processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return _postprocess(text)

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """

        audio = _load_audio_bytes(audio_bytes, target_sr=16000)

        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.dtype,
        )

        if self.use_mixed_beam:
            # First pass: beam 1. Better for Chinese on your validation.
            text_b1 = self._decode(input_features, beams=1)

            # If it looks Chinese, keep beam 1.
            if _count_cjk(text_b1) >= 2:
                return text_b1

            # Otherwise use beam 3, which helped Malay/Tamil/English slightly.
            return self._decode(input_features, beams=3)

        return self._decode(input_features, beams=self.num_beams)