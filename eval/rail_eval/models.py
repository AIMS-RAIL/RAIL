"""Hugging Face `transformers` model adapters.

An adapter turns (audio files, prompt) into generated text. Models are loaded with
`from_pretrained` from a Hub id or a local checkpoint directory:

* `qwen2_audio`  Qwen2-Audio-7B(-Instruct): native multi-audio chat template.
* `generic`      any model whose processor supports audio entries in `apply_chat_template`
                 (e.g. Qwen2.5-Omni, Gemma-3n, Voxtral, Audio Flamingo 3 in recent transformers).
* `auto`         picks `qwen2_audio` for `model_type == "qwen2_audio"`, otherwise `generic`.

Add a model family by subclassing `AudioLM` and registering it in `ADAPTERS`.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

MERGE_GAP_S = 0.3


def load_audio(path: str, sr: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def merge_clips(clips: List[np.ndarray], sr: int, gap_s: float = MERGE_GAP_S) -> np.ndarray:
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    parts: List[np.ndarray] = []
    for i, clip in enumerate(clips):
        if i:
            parts.append(gap)
        parts.append(clip)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def merged_prompt_note(n: int) -> str:
    return (
        f"The provided input audio is a concatenation of {n} clips in order, "
        f"separated by {MERGE_GAP_S:.1f} s of silence.\n"
    )


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
            "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[name]


def _from_pretrained(cls, path: str, dtype, trust_remote_code: bool, attn_implementation: Optional[str]):
    kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return cls.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


class AudioLM:
    name = "base"

    def __init__(self, model_path: str, dtype: str = "bfloat16", device: str = "cuda",
                 trust_remote_code: bool = False, attn_implementation: Optional[str] = None,
                 merge_multi_audio: bool = False):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.merge_multi_audio = merge_multi_audio
        self.processor = None
        self.model = None

    # -- to implement -----------------------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def _inputs(self, clips: List[np.ndarray], prompt: str):
        raise NotImplementedError

    # -- shared -----------------------------------------------------------------------------
    @property
    def sampling_rate(self) -> int:
        fe = getattr(self.processor, "feature_extractor", None)
        return int(getattr(fe, "sampling_rate", 16000) or 16000)

    @property
    def max_clip_seconds(self) -> Optional[float]:
        fe = getattr(self.processor, "feature_extractor", None)
        n = getattr(fe, "n_samples", None)
        return float(n) / self.sampling_rate if n else None

    def describe(self) -> Dict[str, object]:
        import torch
        import transformers

        return {
            "adapter": self.name,
            "model_path": self.model_path,
            "model_class": type(self.model).__name__ if self.model is not None else None,
            "dtype": self.dtype,
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "sampling_rate": self.sampling_rate,
            "max_clip_seconds": self.max_clip_seconds,
            "merge_multi_audio": self.merge_multi_audio,
        }

    def generate(self, audio_files: List[str], prompt: str, max_new_tokens: int) -> Dict[str, object]:
        import torch

        sr = self.sampling_rate
        clips = [load_audio(p, sr) for p in audio_files]
        limit = self.max_clip_seconds
        truncated = sum(1 for c in clips if limit and len(c) / sr > limit + 1e-3)
        if self.merge_multi_audio and len(clips) > 1:
            prompt = merged_prompt_note(len(clips)) + prompt
            clips = [merge_clips(clips, sr)]
            truncated = int(bool(limit and len(clips[0]) / sr > limit + 1e-3))

        inputs = self._inputs(clips, prompt).to(self.device)
        dtype = _torch_dtype(self.dtype)
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(dtype)
        n_in = int(inputs["input_ids"].shape[-1])
        start = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, tuple):  # omni models may also return audio
            out = out[0]
        new_tokens = out[:, n_in:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return {
            "response": text,
            "n_input_tokens": n_in,
            "n_output_tokens": int(new_tokens.shape[-1]),
            "hit_max_new_tokens": int(new_tokens.shape[-1]) >= max_new_tokens,
            "gen_seconds": round(time.time() - start, 3),
            "n_clips_truncated": truncated,
        }


class Qwen2AudioLM(AudioLM):
    name = "qwen2_audio"

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.model = _from_pretrained(Qwen2AudioForConditionalGeneration, self.model_path,
                                      _torch_dtype(self.dtype), self.trust_remote_code,
                                      self.attn_implementation).to(self.device).eval()

    def _inputs(self, clips: List[np.ndarray], prompt: str):
        content = [{"type": "audio", "audio_url": f"clip_{i}"} for i in range(len(clips))]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        return self.processor(text=text, audio=clips, sampling_rate=self.sampling_rate,
                              return_tensors="pt", padding=True)


class GenericChatLM(AudioLM):
    name = "generic"
    MODEL_CLASSES = ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM", "AutoModel")

    def load(self) -> None:
        import transformers
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        errors = []
        for cls_name in self.MODEL_CLASSES:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = _from_pretrained(cls, self.model_path, _torch_dtype(self.dtype),
                                              self.trust_remote_code, self.attn_implementation)
                break
            except (ValueError, KeyError, OSError) as exc:
                errors.append(f"{cls_name}: {exc}")
        if self.model is None:
            raise RuntimeError("no Auto class could load the model:\n" + "\n".join(errors))
        self.model = self.model.to(self.device).eval()

    def _inputs(self, clips: List[np.ndarray], prompt: str):
        content = [{"type": "audio", "audio": clip} for clip in clips]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        try:
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", sampling_rate=self.sampling_rate,
            )
        except (TypeError, ValueError, KeyError):
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            return self.processor(text=text, audio=clips, sampling_rate=self.sampling_rate,
                                  return_tensors="pt", padding=True)


ADAPTERS = {"qwen2_audio": Qwen2AudioLM, "generic": GenericChatLM}


def build_model(model_path: str, adapter: str = "auto", **kwargs) -> AudioLM:
    if adapter == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=kwargs.get("trust_remote_code", False))
        adapter = "qwen2_audio" if getattr(cfg, "model_type", "") == "qwen2_audio" else "generic"
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}; choose from {sorted(ADAPTERS)} or 'auto'")
    model = ADAPTERS[adapter](model_path, **kwargs)
    model.load()
    return model
