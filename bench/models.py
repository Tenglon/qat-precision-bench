"""Five ~1B-class models, one per modality, built from vendored HF configs.

Weights are RANDOM-INIT (`from_config`): throughput depends only on the
architecture (GEMM shapes / layer counts), not on weight values, so this
benchmarks identically to the pretrained checkpoints while avoiding any model
download on the air-gapped cluster.

Each builder returns:
  build()            -> nn.Module on CUDA (fp32)
  make_batch(bs, train) -> dict of CUDA tensors for model(**batch)
  tokens_per_sample  -> int used for tokens/s accounting (0 = report samples/s)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

CFG_DIR = os.environ.get(
    "QATBENCH_CFG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"),
)

TINY = os.environ.get("QATBENCH_TINY") == "1"   # 2-layer CPU smoke-test mode
DEVICE = "cpu" if TINY else "cuda"

_TINY_OVERRIDES = {
    "num_hidden_layers": 2, "encoder_layers": 2, "decoder_layers": 2,
}


def _load_cfg(fname, **overrides):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.path.join(CFG_DIR, fname))
    # transformers v5 nests some configs (e.g. Qwen2VL text_config); apply
    # every override to the top level AND any sub-config that has the field.
    subs = [cfg] + [getattr(cfg, s) for s in ("text_config", "vision_config")
                    if getattr(cfg, s, None) is not None]
    for k, v in overrides.items():
        hit = False
        for c in subs:
            if hasattr(c, k):
                setattr(c, k, v)
                hit = True
        if not hit:
            setattr(cfg, k, v)
    if TINY:
        for c in subs:
            for k, v in _TINY_OVERRIDES.items():
                if hasattr(c, k):
                    setattr(c, k, v)
            if hasattr(c, "depth"):
                c.depth = 2
    return cfg


def _field(cfg, name):
    """Read a config field from the top level or the nested text_config."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return getattr(cfg.text_config, name)


def _build_on_gpu(cls, cfg, attn="sdpa"):
    torch.manual_seed(17)
    with torch.device(DEVICE):
        if hasattr(cls, "from_config"):        # Auto* classes
            model = cls.from_config(cfg, attn_implementation=attn)
        else:                                   # concrete model classes
            cfg._attn_implementation = attn
            model = cls(cfg)
    model = model.float().to(DEVICE)
    return model


# ---------------------------------------------------------------- language

def _lang_family(cfg_file, label, desc, train_bs=8, infer_bs=16):
    ov = {"use_cache": False}
    if TINY:
        ov["vocab_size"] = 8192
    cfg = _load_cfg(cfg_file, **ov)
    seq = 128 if TINY else 1024

    def build():
        from transformers import AutoModelForCausalLM
        m = _build_on_gpu(AutoModelForCausalLM, cfg)
        m.generation_config.eos_token_id = None  # random weights: never early-stop
        m.generation_config.pad_token_id = 0
        return m

    def make_batch(bs, train):
        ids = torch.randint(0, cfg.vocab_size, (bs, seq), device=DEVICE)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if train:
            batch["labels"] = ids.clone()
        return batch

    return Spec(label, desc, build, make_batch,
                tokens_per_sample=seq, train_bs=train_bs, infer_bs=infer_bs,
                supports_decode=True)


def _lang():
    return _lang_family("lang_qwen2.5-1.5b.json", "lang",
                        "Qwen2.5-1.5B (random init)")


def _lang05():
    return _lang_family("lang_qwen2.5-0.5b.json", "lang05",
                        "Qwen2.5-0.5B (random init)", train_bs=16)


def _lang3():
    return _lang_family("lang_qwen2.5-3b.json", "lang3",
                        "Qwen2.5-3B (random init)", train_bs=4)


def _lang7():
    # NOTE: full AdamW training (16 B/param of fp32 state) cannot fit 7B on a
    # 64 GB GPU in ANY compute precision — train records will show OOM, which
    # is itself the finding. Inference/decode run fine.
    return _lang_family("lang_qwen2.5-7b.json", "lang7",
                        "Qwen2.5-7B (random init)", train_bs=1)


# ---------------------------------------------------------------- image

def _image():
    cfg = _load_cfg("image_dinov2-giant.json", num_labels=1000)

    def build():
        from transformers import AutoModelForImageClassification
        return _build_on_gpu(AutoModelForImageClassification, cfg)

    def make_batch(bs, train):
        batch = {"pixel_values": torch.randn(bs, 3, 224, 224, device=DEVICE)}
        if train:
            batch["labels"] = torch.randint(0, 1000, (bs,), device=DEVICE)
        return batch

    return Spec("image", "DINOv2-giant 1.1B + cls head (random init)", build,
                make_batch, tokens_per_sample=0, train_bs=16, infer_bs=64)


# ---------------------------------------------------------------- video

def _video():
    cfg = _load_cfg("video_videomae-huge.json")

    def build():
        from transformers import AutoModelForVideoClassification
        return _build_on_gpu(AutoModelForVideoClassification, cfg)

    def make_batch(bs, train):
        batch = {"pixel_values": torch.randn(bs, 16, 3, 224, 224, device=DEVICE)}
        if train:
            batch["labels"] = torch.randint(0, cfg.num_labels, (bs,), device=DEVICE)
        return batch

    return Spec("video", "VideoMAE-huge 0.6B, 16 frames (random init)", build,
                make_batch, tokens_per_sample=0, train_bs=4, infer_bs=16)


# ---------------------------------------------------------------- audio

def _audio():
    ov = {"use_cache": False}
    if TINY:
        ov.update(vocab_size=8192, decoder_start_token_id=1, pad_token_id=0,
                  bos_token_id=1, eos_token_id=2)
    cfg = _load_cfg("audio_whisper-large-v3.json", **ov)
    dec_len = 32 if TINY else 128

    def build():
        from transformers import AutoModelForSpeechSeq2Seq
        return _build_on_gpu(AutoModelForSpeechSeq2Seq, cfg)

    def make_batch(bs, train):
        feats = torch.randn(bs, cfg.num_mel_bins, 3000, device=DEVICE)
        if train:
            labels = torch.randint(0, cfg.vocab_size, (bs, dec_len), device=DEVICE)
            return {"input_features": feats, "labels": labels}
        dec = torch.randint(0, cfg.vocab_size, (bs, dec_len), device=DEVICE)
        return {"input_features": feats, "decoder_input_ids": dec}

    return Spec("audio", "Whisper-large-v3 1.5B (random init)", build, make_batch,
                tokens_per_sample=0, train_bs=4, infer_bs=8)


# ---------------------------------------------------------------- multimodal

def _mm():
    ov = {"use_cache": False}
    if TINY:
        ov.update(vocab_size=8192, vision_start_token_id=8000,
                  vision_end_token_id=8001, vision_token_id=8002,
                  image_token_id=8003, video_token_id=8004,
                  bos_token_id=1, eos_token_id=2)
    cfg = _load_cfg("mm_qwen2-vl-2b.json", **ov)
    # one 224x224 image -> vision grid (t=1, h=16, w=16) = 256 patches,
    # merged 2x2 -> 64 image tokens in the LM sequence
    grid = (1, 16, 16)
    n_patch = grid[0] * grid[1] * grid[2]
    n_img_tok = n_patch // 4
    patch_dim = 3 * 2 * 14 * 14  # channels * temporal_patch * patch * patch = 1176
    txt_len = 64
    txt_hi = min(151000, _field(cfg, "vocab_size") - 2000)  # clear of special ids

    def build():
        from transformers import Qwen2VLForConditionalGeneration
        return _build_on_gpu(Qwen2VLForConditionalGeneration, cfg)

    def make_batch(bs, train):
        vs, ve, ip = (_field(cfg, "vision_start_token_id"),
                      _field(cfg, "vision_end_token_id"),
                      _field(cfg, "image_token_id"))
        rows = []
        for _ in range(bs):
            txt = torch.randint(0, txt_hi, (txt_len,))
            row = torch.cat([torch.tensor([vs]),
                             torch.full((n_img_tok,), ip),
                             torch.tensor([ve]), txt])
            rows.append(row)
        ids = torch.stack(rows).to(DEVICE)
        mm_types = torch.zeros_like(ids)
        mm_types[:, 1:1 + n_img_tok] = 1        # image-token span for M-RoPE
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(bs * n_patch, patch_dim, device=DEVICE),
            "image_grid_thw": torch.tensor([list(grid)] * bs, device=DEVICE),
            "mm_token_type_ids": mm_types,
        }
        if train:
            labels = ids.clone()
            labels[:, : n_img_tok + 2] = -100  # don't compute LM loss on vision span
            batch["labels"] = labels
        return batch

    return Spec("mm", "Qwen2-VL-2B (random init), 224px image + 64 text tokens",
                build, make_batch,
                tokens_per_sample=n_img_tok + 2 + txt_len, train_bs=4, infer_bs=8)


# ----------------------------------------------------------------

@dataclass
class Spec:
    name: str
    desc: str
    build: Callable
    make_batch: Callable
    tokens_per_sample: int
    train_bs: int
    infer_bs: int
    supports_decode: bool = False


BUILDERS = {"lang": _lang, "image": _image, "video": _video,
            "audio": _audio, "mm": _mm,
            "lang05": _lang05, "lang3": _lang3, "lang7": _lang7}


def get_spec(name: str) -> Spec:
    return BUILDERS[name]()
