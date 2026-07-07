#!/usr/bin/env python3
"""中文流式 SFT 数据构造 —— 在上游 cons_online_data.py 之上做四件事, 不改上游一行代码:

  1. [复用] step1  采样每轮前导/尾部噪声(上游原函数)
  2. [重写] step2  拼接长波形, 新增三项"响应延迟"优化:
       a) 语音尾部静音修剪 (librosa.effects.trim, top_db=40)
          —— TTS wav 尾部常带静音, 不剪的话模型学会"多等一会儿再答"。
       b) 块边界对齐: 微调前导噪声长度, 使每轮语音结束帧 %10 == 8
          —— 回复决策点固定在语音结束后 80ms(否则平均 ~220ms、最差 400ms)。
             对应论文 half-chunk align (δ=200ms) 思路。
       c) 噪声段边缘 20ms 淡入淡出(论文 fade window ω=20ms)
          —— 消除硬拼接咔哒声, 防止模型拿爆音当"说完了"的线索。
  3. [复用] step3  Qwen2.5-Omni audio_tower 抽特征(上游原函数)
  4. [复用+补丁] step4  铺 token; 运行前把上游 DEFAULT_SYSTEM_PROMPT 换成中文
     (来自 zh_config.ZH_SYSTEM_PROMPT, 推理端 infer_online_zh.py 用同一常量)。

  最后做产物校验: 超长样本过滤(防止 fill_in_audio_feature 越界崩溃)、
  特征行数 == 10×chunk 数、决策延迟统计。

用法(在仓库根或任意目录均可):
  python zh_finetune/cons_online_data_zh.py \
      --input online_input.jsonl --checkpoint-dir checkpoints/audiointeraction \
      --work-dir runtime/work --out runtime/train_jsonl/train.jsonl \
      --noise-dir runtime/noise --max-seq 4096
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from zh_finetune.zh_config import (  # noqa: E402
    ALIGN_TARGET_MOD, CROSSFADE_MS, TRIM_MIN_KEEP_S, TRIM_TOP_DB, ZH_SYSTEM_PROMPT,
)


def _lazy_imports():
    """重依赖延迟加载(torch/transformers/whisper), 让 --help 秒开。"""
    global up, whisper, sf, librosa, resolve_checkpoint_paths, SAMPLES_PER_FRAME
    import librosa            # noqa: F401
    import soundfile as sf    # noqa: F401
    import whisper            # noqa: F401

    # --- 上游 bug 修补(不改上游文件): cons_online_data.py 从 generate.base 导入
    # resolve_checkpoint_paths, 但该函数实际定义在仓库根的 utils.py(发布版遗漏)。
    # 先把符号注入 base 模块, 让上游的 from-import 能解析。
    import src.audiointeraction.generate.base as _base
    from utils import resolve_checkpoint_paths  # 仓库根 utils.py
    if not hasattr(_base, "resolve_checkpoint_paths"):
        _base.resolve_checkpoint_paths = resolve_checkpoint_paths

    import src.audiointeraction.dataset.cons_online_data as up
    from src.audiointeraction.dataset.utils.load_audio import SAMPLES_PER_FRAME


# ---------- step2 重写: trim + 对齐 + fade ----------

def _trim_speech(src_path, dst_path, top_db):
    """16k 单声道加载 + 首尾静音修剪; 过度修剪则回退原音频。返回 dst_path。"""
    y, _ = librosa.load(src_path, sr=16000, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < int(TRIM_MIN_KEEP_S * 16000):
        yt = y
    sf.write(dst_path, yt.astype(np.float32), 16000)
    return dst_path


def _load_noise_wrap(noise_path, start_s, n_samples):
    """取 n_samples 噪声; 文件不够长就首尾相接平铺(比上游 raise 更稳)。"""
    full = whisper.load_audio(noise_path, sr=16000)
    if len(full) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    start = int(round(start_s * 16000)) % len(full)
    idx = (start + np.arange(n_samples)) % len(full)
    return full[idx].astype(np.float32)


def _fade_edges(seg, fade_samples):
    """就地对段首淡入、段尾淡出(线性), 不改变长度。"""
    n = len(seg)
    if n == 0:
        return seg
    f = min(fade_samples, n // 2)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        seg[:f] *= ramp
        seg[-f:] *= ramp[::-1]
    return seg


def step2_concat_audio_zh(input_jsonl, output_jsonl, wavs_dir, trimmed_dir, *,
                          chunk_size, align, trim_top_db, seed=1337):
    """上游 step2 的替代实现: [噪声→语音]×N→尾噪声, 带 trim/对齐/fade。

    帧数计算完全复用上游 `_load_audio_aligned`(对修剪后的 wav 调用),
    与 step3 特征提取的卷积长度公式保持一致。
    """
    import random
    random.seed(seed)  # 上游 _load_audio_aligned 超长裁剪时用 random, 保证可复现
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    fade_n = int(CROSSFADE_MS / 1000 * 16000)

    delays_ms = []
    from tqdm import tqdm
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin.readlines(), desc="step2(zh)"):
            rec = json.loads(line)
            try:
                segments = []
                cum = 0
                for k, t in enumerate(rec["turns"]):
                    trimmed = _trim_speech(
                        t["audio_path"],
                        os.path.join(trimmed_dir, f"{rec['idx']}_{k}.wav"),
                        trim_top_db,
                    )
                    seg, n_frames = up._load_audio_aligned(trimmed)

                    lead = t["leading_silence_frames"]
                    if align:
                        # 补齐前导噪声, 使本轮结束帧 % chunk == ALIGN_TARGET_MOD
                        lead += (ALIGN_TARGET_MOD - (cum + lead + n_frames) % chunk_size) % chunk_size
                    noise = _load_noise_wrap(
                        t["leading_noise_path"], t["leading_noise_start_s"],
                        lead * SAMPLES_PER_FRAME,
                    )
                    _fade_edges(noise, fade_n)
                    segments += [noise, seg]

                    t["leading_silence_frames"] = lead
                    t["audio_frames"] = n_frames
                    cum += lead + n_frames
                    # 该轮回复决策点相对语音结束的滞后
                    delays_ms.append((chunk_size - cum % chunk_size) % chunk_size * 40 or chunk_size * 40)

                # 尾部静音: 与上游同一套取整逻辑, 使总帧数落在 chunk 边界
                tail = rec["tail_silence_frames"] - (cum + rec["tail_silence_frames"]) % chunk_size
                if tail < 0:
                    tail = (chunk_size - (cum % chunk_size)) % chunk_size
                tail_noise = _load_noise_wrap(
                    rec["tail_noise_path"], rec["tail_noise_start_s"],
                    tail * SAMPLES_PER_FRAME,
                )
                _fade_edges(tail_noise, fade_n)
                segments.append(tail_noise)
                rec["tail_silence_frames_actual"] = tail

                wav_path = os.path.join(wavs_dir, f"{rec['idx']}.wav")
                up._write_wav(wav_path, np.concatenate(segments))
                rec["concat_wav_path"] = wav_path
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step2zh idx {rec.get('idx')}] {type(e).__name__}: {e}")

    if delays_ms:
        print(f"[延迟] 回复决策点滞后于语音结束: mean={np.mean(delays_ms):.0f}ms "
              f"max={np.max(delays_ms):.0f}ms (align={'on' if align else 'off'})")


# ---------- 产物校验 ----------

def verify_and_filter(train_jsonl, *, max_seq, chunk_size, check_feats_n=8):
    """1) 丢弃超过 max_seq 的样本(否则训练期 fill_in_audio_feature 越界报错);
       2) 抽查前 N 条: AudioFeat 行数 == chunk_size × len(audio_pos);
       3) 抽查 labels: 存在被监督的回复 token。"""
    import torch

    kept, dropped = [], 0
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["input_ids"]) > max_seq:
                dropped += 1
                continue
            kept.append(line)
    if dropped:
        with open(train_jsonl, "w", encoding="utf-8") as f:
            f.writelines(kept)
    print(f"[校验] 样本 {len(kept)} 条保留, {dropped} 条超长(> {max_seq})被过滤")
    if not kept:
        sys.exit("[校验] 没有可用样本, 中止")

    bad = 0
    for line in kept[:check_feats_n]:
        rec = json.loads(line)
        feat = torch.load(os.path.join(rec["pt_path_dir"], "AudioFeat.pt"), map_location="cpu")
        want = chunk_size * len(rec["audio_pos"])
        if feat.shape[0] != want:
            bad += 1
            print(f"[校验][idx {rec['idx']}] 特征行数 {feat.shape[0]} != chunk×pos {want} !!")
        n_sup = sum(1 for x in rec["labels"] if x != -100)
        if n_sup == 0:
            bad += 1
            print(f"[校验][idx {rec['idx']}] labels 全被 mask !!")
    print(f"[校验] 抽查 {min(check_feats_n, len(kept))} 条特征对齐: "
          + ("全部通过 ✓" if bad == 0 else f"{bad} 条异常 ✗"))
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="中文流式 SFT 数据构造(上游 4 步 + 延迟优化)")
    ap.add_argument("--input", required=True, help="convert_to_online_input.py 的输出 jsonl")
    ap.add_argument("--checkpoint-dir", required=True, help="HF snapshot 根(tokenizer/qwenOmni/tower)")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True, help="最终训练 jsonl")
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--no-align", action="store_true", help="关闭块边界对齐优化")
    ap.add_argument("--trim-top-db", type=float, default=TRIM_TOP_DB)
    ap.add_argument("--min-noise-len", type=int, default=20)
    ap.add_argument("--max-noise-len", type=int, default=60)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-steps", default="", help="逗号分隔跳过的步骤, 如 1,2,3(断点续跑)")
    args = ap.parse_args()

    _lazy_imports()
    skip = {s.strip() for s in args.skip_steps.split(",") if s.strip()}

    wd = os.path.abspath(args.work_dir)
    os.makedirs(wd, exist_ok=True)
    s1, s2, s3 = (os.path.join(wd, f"step{i}.jsonl") for i in (1, 2, 3))
    wavs, trimmed, feats = (os.path.join(wd, d) for d in ("wavs", "trimmed", "features"))

    tokenizer_dir, _, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(args.checkpoint_dir)

    if "1" not in skip:
        up.step1_sample_silence(
            args.input, s1, noise_dir=args.noise_dir,
            min_noise_len=args.min_noise_len, max_noise_len=args.max_noise_len,
            chunk_size=args.chunk_size, seed=args.seed,
        )
    if "2" not in skip:
        step2_concat_audio_zh(
            s1, s2, wavs, trimmed,
            chunk_size=args.chunk_size, align=not args.no_align,
            trim_top_db=args.trim_top_db, seed=args.seed,
        )
    if "3" not in skip:
        up.step3_extract_features(
            s2, s3, feats,
            qwen_omni_ckpt=qwen_omni_ckpt, audio_tower_ckpt=audio_tower_ckpt,
            device=args.device,
        )
    if "4" not in skip:
        # 核心补丁: 中文系统提示词(上游 step4 运行时读模块全局, 补丁即生效)
        up.DEFAULT_SYSTEM_PROMPT = ZH_SYSTEM_PROMPT
        print(f"[step4] 使用中文 system prompt: {ZH_SYSTEM_PROMPT!r}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        up.step4_build_tokens(s3, args.out, tokenizer_dir=tokenizer_dir, chunk_size=args.chunk_size)

    verify_and_filter(args.out, max_seq=args.max_seq, chunk_size=args.chunk_size)
    print(f"\n[完成] 训练数据: {args.out}")


if __name__ == "__main__":
    main()
