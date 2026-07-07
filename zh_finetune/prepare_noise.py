#!/usr/bin/env python3
"""准备背景噪声库(step1 的空档填充用)。

优先级:
  1. 目标目录已有足量 wav → 直接用(幂等)。
  2. sparse-clone 微软 MS-SNSD 的 noise_train(DNS-Challenge 同源噪声, ~360MB,
     128 个 16kHz wav: Babble/Cafe/AirConditioner/Traffic...)。
  3. 网络不可用 → 合成有色噪声兜底(仅保证流程可跑, 建议真实训练用真噪声)。

用法: python zh_finetune/prepare_noise.py --dest runtime/noise [--min-files 20]
"""
import argparse
import glob
import os
import subprocess
import sys


def count_wavs(d):
    return len(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))


def try_ms_snsd(dest):
    tgt = os.path.join(dest, "MS-SNSD")
    if count_wavs(tgt) >= 20:
        return True
    os.makedirs(dest, exist_ok=True)
    try:
        if not os.path.isdir(os.path.join(tgt, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                 "https://github.com/microsoft/MS-SNSD.git", tgt],
                check=True, timeout=600,
            )
        subprocess.run(["git", "-C", tgt, "sparse-checkout", "set", "noise_train"],
                       check=True, timeout=600)
        return count_wavs(tgt) >= 20
    except Exception as e:
        print(f"[noise] MS-SNSD 下载失败: {e}")
        return False


def synth_fallback(dest, n_files=24, seconds=60):
    """合成有色噪声(粉噪+低频起伏), 16kHz mono wav。仅兜底。"""
    import numpy as np
    import soundfile as sf

    out = os.path.join(dest, "synth")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1337)
    n = seconds * 16000
    for i in range(n_files):
        white = rng.standard_normal(n).astype(np.float32)
        spec = np.fft.rfft(white)
        f = np.maximum(np.fft.rfftfreq(n, 1 / 16000), 1.0)
        pink = np.fft.irfft(spec / np.sqrt(f), n).astype(np.float32)
        env = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * np.arange(n) / 16000)
        y = pink * env
        y = 0.05 * y / (np.abs(y).max() + 1e-8)
        sf.write(os.path.join(out, f"synthnoise_{i:02d}.wav"), y, 16000)
    print(f"[noise] 已合成 {n_files} 个兜底噪声 -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--min-files", type=int, default=20)
    args = ap.parse_args()

    if count_wavs(args.dest) >= args.min_files:
        print(f"[noise] 已有 {count_wavs(args.dest)} 个 wav, 跳过")
        return
    if try_ms_snsd(args.dest):
        print(f"[noise] MS-SNSD 就绪, 共 {count_wavs(args.dest)} 个 wav")
        return
    print("[noise] 转合成兜底(建议真实训练前换成 MS-SNSD/DNS 真噪声)")
    synth_fallback(args.dest)
    if count_wavs(args.dest) < args.min_files:
        sys.exit("[noise] 兜底失败")


if __name__ == "__main__":
    main()
