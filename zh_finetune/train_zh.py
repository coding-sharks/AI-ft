#!/usr/bin/env python3
"""训练启动器 = 上游 train.py + (可选) GQA flash-attn 补丁 + (可选) 单卡激活重算补丁。
不改上游任何文件。

用法(等价于直接跑上游 train.py, 只是多了补丁):
    python zh_finetune/train_zh.py --config zh_finetune/config_zh_smoke.yaml
环境变量:
    ZH_USE_FLASH=0   禁用 flash 补丁(默认 1: flash-attn 可用则启用, 不可用自动回退)
    ZH_GRAD_CKPT     激活重算(gradient checkpointing):
                     1=强开, 0=强关, 缺省=AI_DEVICES==1 时自动开。
                     背景: 上游 train.py 只在多卡(FSDP)时启用激活重算, 单卡什么都没有 →
                     3B 全参 seq4096 单卡必 OOM。本补丁给 Block.forward 包一层
                     torch.utils.checkpoint, 单卡显存 100G+ → ~70G, 代价 ~+30% 步时。
                     多卡时保持关闭(FSDP 自带, 双重 ckpt 只会变慢)。
"""
import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _enable_grad_ckpt():
    """把 Block.forward 包上 torch.utils.checkpoint(仅训练态生效)。幂等。"""
    import torch
    from torch.utils.checkpoint import checkpoint
    from src.audiointeraction.model import Block

    if getattr(Block, "_zh_ckpt", False):
        return
    _orig = Block.forward

    def _ckpt_forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(_orig, self, *args, use_reentrant=False, **kwargs)
        return _orig(self, *args, **kwargs)

    Block.forward = _ckpt_forward
    Block._zh_ckpt = True


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

    ckpt_env = os.environ.get("ZH_GRAD_CKPT", "")
    if ckpt_env == "1" or (ckpt_env == "" and os.environ.get("AI_DEVICES", "") == "1"):
        _enable_grad_ckpt()
        print("[ckpt] 单卡激活重算补丁: ✅ 已启用(显存↓↓, 步时约 +30%)")
    else:
        print("[ckpt] 激活重算补丁: 关(多卡走 FSDP 自带的重算)")

    from src.audiointeraction.finetune.train import load_config, setup
    setup(load_config(args.config))


if __name__ == "__main__":
    main()
