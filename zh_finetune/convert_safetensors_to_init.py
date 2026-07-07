#!/usr/bin/env python3
"""把 HF 仓库的分片 safetensors 合并成 train.py 需要的 init.pt。

作者的 utils.load_model 直接把这些分片 strict=True 加载进 litgpt GPT,
说明 key 即 litgpt 原生格式(transformer.* / lm_head.*), 因此这里只需
"合并 → (可选校验) → torch.save"。

用法:
  python zh_finetune/convert_safetensors_to_init.py \
      --ckpt-dir checkpoints/audiointeraction --out runtime/init.pt --verify
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="meta device 上建 GPT, 对照 key/shape(需完整训练依赖)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file

    ckpt = Path(args.ckpt_dir)
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.is_file():
        sys.exit(f"缺少 {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    print(f"合并 {len(shards)} 个分片 ...")

    sd = {}
    for s in shards:
        sd.update(load_file(str(ckpt / s), device="cpu"))
    n_params = sum(v.numel() for v in sd.values())
    heads = sorted({k.split(".")[0] for k in sd})
    print(f"张量 {len(sd)} 个, 参数量 {n_params/1e9:.3f}B, 顶层前缀: {heads}")

    if not any(k.startswith("transformer.") for k in sd):
        sys.exit("!! key 不是 litgpt 格式(没有 transformer.*), 需要人工确认映射, 中止")

    if args.verify:
        from src.audiointeraction.model import GPT, Config
        cfg = Config.from_file(ckpt / "model_config.yaml")
        with torch.device("meta"):
            model = GPT(cfg)
        ref = model.state_dict()
        missing = [k for k in ref if k not in sd]
        unexpected = [k for k in sd if k not in ref]
        shape_bad = [k for k in ref if k in sd and ref[k].shape != sd[k].shape]
        print(f"校验: missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_bad)}")
        for name, lst in (("missing", missing), ("unexpected", unexpected), ("shape", shape_bad)):
            for k in lst[:5]:
                print(f"  [{name}] {k}")
        if missing or shape_bad:
            sys.exit("!! 校验未通过, 中止")
        print(f"模型架构: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
              f"n_query_groups={cfg.n_query_groups} padded_vocab={cfg.padded_vocab_size}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"已保存 init.pt -> {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
