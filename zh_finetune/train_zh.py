#!/usr/bin/env python3
"""训练启动器 = 上游 train.py + (可选) GQA flash-attn 补丁。不改上游任何文件。

用法(等价于直接跑上游 train.py, 只是多了补丁):
    python zh_finetune/train_zh.py --config zh_finetune/config_zh_smoke.yaml
环境变量:
    ZH_USE_FLASH=0   禁用 flash 补丁(默认 1: flash-attn 可用则启用, 不可用自动回退)
"""
import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("ZH_USE_FLASH", "1") == "1":
        from zh_finetune.flash_patch import apply_patch
        enabled = apply_patch()
        print(f"[flash] GQA flash-attn 补丁: {'✅ 已启用' if enabled else '不可用 → 回退 SDPA(功能等价)'}")
    else:
        print("[flash] ZH_USE_FLASH=0 → 使用 SDPA")

    from src.audiointeraction.finetune.train import load_config, setup
    setup(load_config(args.config))


if __name__ == "__main__":
    main()
