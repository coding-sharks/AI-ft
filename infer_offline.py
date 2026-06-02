import json
from pathlib import Path
from typing import List, Optional

import lightning as L
import torch

from src.audiointeraction.dataset.TOKENS import ENGLISH, ONLINE, SYSTEM, TEXT_BEGIN, TEXT_END
from src.audiointeraction.generate.base import streaming_generate
from utils import (
    load_audio_encoder, load_model, resolve_checkpoint_paths, set_seed, get_best_device
)
from src.audiointeraction.tokenizer import Tokenizer
from src.audiointeraction.utils import get_default_supported_precision


SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)


def run_inference(
    *,
    checkpoint_dir: str,
    rounds: int = 10,
    audio_paths: Optional[List[str]] = None,
    seed: int = 1337,
    max_new_tokens: int = 4096,
    device: str = "cuda:0",
):
    """End-to-end: build fabric, load model + audio encoder, run streaming_generate.

    If `audio_paths` is given, runs one round per path non-interactively
    (offline mode). Otherwise prompts stdin each round (online mode).
    """
    if not checkpoint_dir:
        raise RuntimeError("`checkpoint_dir` is empty — set it before calling run_inference().")
    model_config_dir, trained_checkpoint, qwen_omni_ckpt, audio_tower_ckpt = \
        resolve_checkpoint_paths(checkpoint_dir)

    set_seed(seed)
    fabric = L.Fabric(
        devices=1, num_nodes=1, strategy="auto",
        precision=get_default_supported_precision(training=False),
        loggers="tensorboard",
    )
    model = load_model(fabric, model_config_dir, trained_checkpoint).to(device)
    audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
    tokenizer = Tokenizer(model_config_dir)

    system_ids = tokenizer.encode(SYSTEM_PROMPT).cpu().tolist()
    prefix_ids = torch.LongTensor(
        [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    ).to(model.device)

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    model.eval()
    try:
        with torch.inference_mode():
            return streaming_generate(
                model, audio_encoder, tokenizer, prefix_ids,
                rounds=rounds, audio_paths=audio_paths,
                max_returned_tokens=max_new_tokens,
                temperature=0.0, top_p=0.0,  # greedy/argmax → deterministic output
            )
    finally:
        model.clear_kv_cache()

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def _load_sequence_json(p: Path) -> List[str]:
    """Read a sequence.json (a plain list of path strings) and resolve relative
    entries against the JSON file's directory."""
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of audio paths in {p}, got {type(data).__name__}.")
    base_dir = p.parent
    return [str(base_dir / a) if not Path(a).is_absolute() else a for a in data]


def load_audio_paths(input_path: str) -> List[str]:
    """Resolve `input_path` into an ordered list of audio files. Layered:

      1. A single audio file        -> processed directly.
      2. A folder with sequence.json -> the order listed in that sequence.json.
      3. A folder of audio files     -> files sorted by name; the order is
                                        printed and you must type 'yes' to go on.

    A .json file may also be passed directly (same as case 2). The files are fed
    sequentially into a single offline session (pseudo-online: context
    accumulates across files).
    """
    p = Path(input_path)

    if p.is_file():
        if p.suffix.lower() == ".json":
            return _load_sequence_json(p)
        return [input_path]

    if p.is_dir():
        seq = p / "sequence.json"
        if seq.is_file():
            return _load_sequence_json(seq)

        files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        if not files:
            raise ValueError(f"No audio files found in {input_path}")
        print(f"No sequence.json in {input_path}; using default order:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f.name}")
        if input("Proceed with this order? type 'yes' to continue: ").strip().lower() not in ("yes", "y"):
            raise SystemExit("Aborted by user.")
        return [str(f) for f in files]

    raise ValueError(f"Input not found: {input_path}")


# A single audio file, or a folder (with a sequence.json, or just loose audio files).
# Bundled samples, e.g. sample/01_count_bark, sample/02_translate, sample/03_cough_music
input_path = "sample/01_count_bark"
audio_paths = load_audio_paths(input_path)

run_inference(checkpoint_dir="./checkpoints",
    audio_paths=audio_paths,
    device=get_best_device(),
)