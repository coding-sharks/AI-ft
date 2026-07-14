#!/usr/bin/env python3
"""SNR 退化确认实验 —— 音频生成端(无需模型, 任何机器可跑)。

目的: 确认"语音上有噪声 → 模型不开口"这条因果链是否成立, 决定 v4 噪声床重训值不值得做。
做法: 给一条【模型已知会开口】的 wav(训练分布样本, 如 data_full/work/wavs/0.wav)
按不同 SNR 叠加连续噪声床, 产出一组 wav; 然后用现有 diag 脚本逐个测
"语音结束点 P(TEXT_BEGIN)", 画出 P 随 SNR 的退化曲线。

判读(测量端见 TODO_v4_noisebed_H20.md):
  - P 在 SNR 10~20dB 就明显崩 → 噪声轴主导实锤, 重建数据(--bed-prob 0.5)重训, 预期收益大;
  - P 到 SNR 5dB 仍稳 → 语音带噪不是瓶颈, 不要盲目重训, 回头查 4A 锚定集与 checkpoint 扫描。

用法(在仓库根):
  python zh_finetune/tools/make_snr_sweep.py \
      --src zh_finetune/runtime/data_full/work/wavs/0.wav \
      --noise-dir zh_finetune/runtime/noise \
      --out /tmp/snr_sweep
产出: /tmp/snr_sweep/snr_clean.wav, snr_20db.wav, snr_15db.wav, snr_10db.wav, snr_05db.wav
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf


def rms_db(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12)


def tile(noise, n):
    reps = int(np.ceil(n / len(noise)))
    return np.tile(noise, reps)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="模型已知会开口的 wav(训练分布样本)")
    ap.add_argument("--noise-dir", required=True, help="噪声库目录(递归找 *.wav)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snrs", default="20,15,10,5", help="逗号分隔的 SNR dB 列表")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    y, sr = sf.read(a.src, dtype="float32")
    assert sr == 16000, f"需要 16k wav, 得到 {sr}"
    if y.ndim > 1:
        y = y.mean(axis=1)

    pool = sorted(glob.glob(os.path.join(a.noise_dir, "**", "*.wav"), recursive=True))
    # 与构造流水线同一原则: 剔除前景人声类
    pool = [p for p in pool if not any(k in os.path.basename(p)
                                       for k in ("Babble", "AirportAnnouncement"))]
    assert pool, f"噪声库为空: {a.noise_dir}"
    noise_path = rng.choice(pool)
    noise, nsr = sf.read(noise_path, dtype="float32")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    assert nsr == 16000, f"噪声需 16k, 得到 {nsr}"
    bed = tile(noise, len(y))

    # 语音响度基准: 用能量最高的 30% 帧估计(整条 wav 大半是安静空档, 直接全局 RMS 会低估)
    frame = 640
    n_fr = len(y) // frame
    fr_rms = np.sqrt(np.mean(np.square(y[: n_fr * frame].reshape(n_fr, frame)), axis=1))
    top = np.sort(fr_rms)[int(n_fr * 0.7):]
    speech_db = 20.0 * np.log10(float(np.mean(top)) + 1e-12)

    sf.write(os.path.join(a.out, "snr_clean.wav"), y, 16000)
    print(f"src={a.src}\nnoise={noise_path}\nspeech_db(top30%帧)={speech_db:.1f}")
    for snr in (float(s) for s in a.snrs.split(",")):
        g = 10.0 ** ((speech_db - snr - rms_db(bed)) / 20.0)
        mix = y + bed * g
        peak = float(np.abs(mix).max())
        if peak > 0.99:
            mix = mix * (0.99 / peak)
        p = os.path.join(a.out, f"snr_{int(snr):02d}db.wav")
        sf.write(p, mix.astype(np.float32), 16000)
        print(f"  → {p} (SNR {snr:.0f} dB)")
    print("完成。接下来用 diag 脚本对每个 wav 测语音结束点 P(TEXT_BEGIN)。")


if __name__ == "__main__":
    main()
