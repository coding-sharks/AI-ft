#!/usr/bin/env python3
"""中文流式推理 —— 加载微调后的 .pt 权重, 用与训练一致的中文 system prompt。

与上游 infer_online.py 的差异:
  * LM 从我们训练产出的 state-dict(.pt)加载(上游只支持 safetensors 分片目录);
  * 前缀用 zh_config.ZH_SYSTEM_PROMPT(必须与训练一致, 单一来源保证);
  * 其余(音频塔/流式状态机/贪心解码)全部复用上游。

用法:
  python zh_finetune/infer_online_zh.py \
      --checkpoint-dir checkpoints/audiointeraction \
      --lm zh_finetune/runtime/train_output/zh-full/final/lit_model.pth \
      [--audio a.wav b.wav]      # 不给则交互式(stdin 里输音频路径)
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from zh_finetune.zh_config import ZH_SYSTEM_PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="HF snapshot 根(tokenizer/tower/config)")
    ap.add_argument("--lm", required=True, help="微调产物 lit_model.pth / init.pt 格式的 state dict")
    ap.add_argument("--audio", nargs="*", default=None)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import lightning as L
    import torch

    from src.audiointeraction.dataset.TOKENS import ENGLISH, ONLINE, SYSTEM, TEXT_BEGIN, TEXT_END
    from src.audiointeraction.generate.base import streaming_generate
    from src.audiointeraction.model import GPT, Config
    from src.audiointeraction.tokenizer import Tokenizer
    from src.audiointeraction.utils import get_default_supported_precision
    from utils import load_audio_encoder, resolve_checkpoint_paths, set_seed

    model_config_dir, _, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(args.checkpoint_dir)
    set_seed(1337)
    fabric = L.Fabric(devices=1, strategy="auto",
                      precision=get_default_supported_precision(training=False))

    config = Config.from_file(Path(model_config_dir) / "model_config.yaml")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    sd = torch.load(args.lm, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model = fabric.setup(model).to(args.device)

    audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, args.device)
    tokenizer = Tokenizer(model_config_dir)

    print(f"[prefix] 中文 system prompt: {ZH_SYSTEM_PROMPT!r}")
    system_ids = tokenizer.encode(ZH_SYSTEM_PROMPT).cpu().tolist()
    prefix_ids = torch.LongTensor(
        [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    ).to(model.device)

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    model.eval()
    try:
        with torch.inference_mode():
            streaming_generate(
                model, audio_encoder, tokenizer, prefix_ids,
                rounds=args.rounds, audio_paths=args.audio,
                max_returned_tokens=args.max_new_tokens,
                temperature=0.0, top_p=0.0,
            )
    finally:
        model.clear_kv_cache()


if __name__ == "__main__":
    main()
