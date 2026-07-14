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
    ALIGN_TARGET_MODS, BED_AMBIENT_EXTRA_DB, BED_PROB, BED_SNR_DB_RANGE, CROSSFADE_MS,
    GAP_FLOOR_DB_RANGE, GAP_NOISE_EXCLUDE_PATTERN, GAP_STYLES, NOISE_ATTEN_DB_RANGE,
    SPEECH_AUG_PROB, TRIM_MIN_KEEP_S, TRIM_TOP_DB, ZH_SYSTEM_PROMPT,
)


def _sf_load_audio(path, sr=16000):
    """soundfile 直读版 load_audio, 语义对齐 whisper.load_audio(解码→单声道→重采样→float32)。

    为什么替换: whisper.load_audio 每次 fork 一个 ffmpeg 子进程; 若 conda 环境在网络盘
    (CephFS 等), 每次 exec 要跨网加载 libav* 动态库, 实测 ~1.4s/次、5 万段音频 = 25 小时,
    且高并发时进程全卡 D 状态。soundfile 进程内直读实测快 3 个数量级。
    等价性: 本 pipeline 语音路径本来就走 librosa/soundfile(load_audio.py:41), ffmpeg 只用于
    16kHz 噪声段与 step3 重读 16kHz 拼接 wav —— 16k→16k 无重采样, int16→float 同为 ÷32768,
    与 ffmpeg 逐位一致; 仅当输入非 16kHz 时用 librosa(soxr)重采样, 与 ffmpeg 差异 ~1e-3 量级。
    """
    import librosa as _lr
    import numpy as _np
    import soundfile as _sf
    data, native_sr = _sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if native_sr != sr:
        data = _lr.resample(data, orig_sr=native_sr, target_sr=sr)
    return _np.ascontiguousarray(data, dtype=_np.float32)


def _lazy_imports(use_sf_audio=True):
    """重依赖延迟加载(torch/transformers/whisper), 让 --help 秒开。"""
    global up, whisper, sf, librosa, resolve_checkpoint_paths, SAMPLES_PER_FRAME
    import librosa            # noqa: F401
    import soundfile as sf    # noqa: F401
    import whisper            # noqa: F401

    # --- 提速补丁(不改上游文件): 进程内把 whisper.load_audio 换成 soundfile 直读,
    # 消除每段音频一次的 ffmpeg fork(上游 step3 与本文件的噪声加载都会自动走新实现)。
    if use_sf_audio:
        whisper.load_audio = _sf_load_audio
        print("[audio] ffmpeg → soundfile 直读补丁: ✅(--no-sf-audio 可回退)")

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

def _augment_speech(y, rng):
    """语音信道模拟(v3): 把干净 TTS 折磨成"浏览器麦克风味"。numpy/librosa 实现。

    子操作独立按概率施加: 随机增益(破 TTS 响度归一)、变速±10%(重采样法, 连带音高
    变化, 模拟语速/声道差异)、轻混响(合成指数衰减 IR)、频谱倾斜+高通(麦克风频响)。
    """
    if rng.random() < 0.8:      # 增益 −8~+4 dB(真实麦常偏小声)
        y = y * (10.0 ** (rng.uniform(-8.0, 4.0) / 20.0))
    if rng.random() < 0.5:      # 变速(+音高)
        rate = rng.uniform(0.9, 1.1)
        y = librosa.resample(y.astype(np.float32), orig_sr=16000,
                             target_sr=int(16000 * rate))
    if rng.random() < 0.4:      # 轻混响
        n = int(rng.uniform(0.05, 0.3) * 16000)
        ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.0 * np.arange(n) / n)
        wet = np.convolve(y, ir)[: len(y)]
        peak_y, peak_w = np.abs(y).max() + 1e-9, np.abs(wet).max() + 1e-9
        y = (1.0 - 0.25) * y + 0.25 * wet * (peak_y / peak_w)
    if rng.random() < 0.5:      # 频谱倾斜 + 高通
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1.0 / 16000)
        tilt = (np.maximum(f, 50.0) / 1000.0) ** rng.uniform(-0.25, 0.25)
        hp = 1.0 / (1.0 + (rng.uniform(60.0, 150.0) / np.maximum(f, 1.0)) ** 2)
        y = np.fft.irfft(Y * tilt * hp, len(y))
    y = np.asarray(y, dtype=np.float32)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y


def _trim_speech(src_path, dst_path, top_db, aug_rng=None):
    """16k 单声道加载 + 首尾静音修剪(+可选信道模拟); 过度修剪则回退原音频。"""
    y, _ = librosa.load(src_path, sr=16000, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < int(TRIM_MIN_KEEP_S * 16000):
        yt = y
    if aug_rng is not None:
        yt = _augment_speech(yt, aug_rng)
    sf.write(dst_path, np.asarray(yt, dtype=np.float32), 16000)
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


def _tile_crossfade(noise_path, start_s, n_samples, fade_n):
    """把噪声文件平铺到 n_samples 长, 接缝处交叉淡化(overlap-add), 消除循环点咔哒。

    与 _load_noise_wrap 的差别: wrap 是模运算硬接缝(空档段短、接缝少, 可接受);
    噪声床横贯整条样本(可达数分钟), 硬接缝会周期性制造"伪边界", 必须交叉淡化。
    """
    full = whisper.load_audio(noise_path, sr=16000)
    if len(full) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    start = int(round(start_s * 16000)) % len(full)
    full = np.concatenate([full[start:], full[:start]]).astype(np.float32)
    if len(full) <= fade_n * 2:
        return _load_noise_wrap(noise_path, start_s, n_samples)
    out = np.zeros(n_samples + len(full), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    tile = full.copy()
    tile[:fade_n] *= ramp
    tile[-fade_n:] *= ramp[::-1]
    pos = 0
    while pos < n_samples:
        out[pos:pos + len(tile)] += tile
        pos += len(tile) - fade_n            # 重叠 fade_n → 接缝处功率近似恒定
    return out[:n_samples]


# step2 多进程 worker 配置(fork 继承; Linux 默认 fork 启动方式)
_W = {}


def _rms_db(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12)


def _step2_one_record(line):
    """处理一条 step1 记录 → (输出行 or None, 延迟列表, 错误信息 or None)。

    可复现性: 每条记录独立播种(seed×大素数+idx), 与 worker 数/调度顺序无关。
    2026-07-09 修复"训后不开口"坍缩:
      * 空档噪声增益控制 —— 衰减到「本对话语音 RMS − U(atten_db)」, 制造真实的
        "说完→变安静"声学边界(原生振幅时噪声仅比语音低 ~3dB, 无边界可学);
      * 对齐目标随机化 —— 结束帧 mod 从恒定 8(80ms 证据/零方差)改为每轮随机
        {4..7}(120~240ms 证据), 保证正样本可学、有方差。
    """
    import random
    rec = json.loads(line)
    try:
        random.seed(_W["seed"] * 1_000_003 + int(rec["idx"]))
        nprng = np.random.default_rng(_W["seed"] * 1_000_003 + int(rec["idx"]))
        chunk_size, fade_n = _W["chunk_size"], _W["fade_n"]
        align_mods = _W["align_mods"]        # tuple 或 None(关闭对齐)
        atten_range = _W["atten_db"]         # (lo,hi) 或 None(关闭增益控制)
        gap_styles = _W["gap_styles"]        # ((name,weight),...) 或 None(仅真噪声)
        aug_prob = _W["speech_aug_prob"]     # 0=关闭信道模拟
        delays = []

        # ---- pass 1: 修剪(+信道模拟)并载入全部语音段, 求本对话语音 RMS 基准 ----
        speeches = []
        for k, t in enumerate(rec["turns"]):
            aug = nprng if (aug_prob > 0 and nprng.random() < aug_prob) else None
            trimmed = _trim_speech(
                t["audio_path"],
                os.path.join(_W["trimmed_dir"], f"{rec['idx']}_{k}.wav"),
                _W["trim_top_db"], aug_rng=aug,
            )
            speeches.append(up._load_audio_aligned(trimmed))
        speech_db = _rms_db(np.concatenate([s for s, _ in speeches]))

        # v4 噪声床: 本条样本是否走"连续噪声床"样式(空档=纯床, 语音上也叠床)
        use_bed = _W["bed_prob"] > 0 and nprng.random() < _W["bed_prob"]

        def _pick_gap_style():
            if not gap_styles:
                return "noise"
            names = [n for n, _ in gap_styles]
            w = np.array([x for _, x in gap_styles], dtype=np.float64)
            return str(nprng.choice(names, p=w / w.sum()))

        def _gap_noise(path, start_s, n_samples):
            """v3 三态空档: 衰减真噪声 / 数字零(浏览器NS) / 极安白噪底。
            床样式下空档=纯零(之后整条叠床, 空档处只剩床——即论文的连续噪声形态)。"""
            if n_samples <= 0:
                return np.zeros(0, dtype=np.float32)
            if use_bed:
                return np.zeros(n_samples, dtype=np.float32)
            style = _pick_gap_style()
            if style == "zero":
                return np.zeros(n_samples, dtype=np.float32)
            if style == "floor":
                amp = 10.0 ** ((speech_db - nprng.uniform(*_W["floor_db"])) / 20.0)
                return _fade_edges(
                    (nprng.standard_normal(n_samples) * amp).astype(np.float32), fade_n)
            noise = _load_noise_wrap(path, start_s, n_samples)
            if atten_range is not None and len(noise):
                target_db = speech_db - random.uniform(*atten_range)
                gain = min(1.0, 10.0 ** ((target_db - _rms_db(noise)) / 20.0))  # 只衰减不放大
                noise = noise * gain
            return _fade_edges(noise, fade_n)

        # ---- pass 2: 拼接 ----
        segments = []
        cum = 0
        for k, t in enumerate(rec["turns"]):
            seg, n_frames = speeches[k]
            lead = t["leading_silence_frames"]
            if align_mods:
                m = random.choice(align_mods)
                lead += (m - (cum + lead + n_frames) % chunk_size) % chunk_size
            segments += [
                _gap_noise(t["leading_noise_path"], t["leading_noise_start_s"],
                           lead * SAMPLES_PER_FRAME),
                seg,
            ]
            t["leading_silence_frames"] = lead
            t["audio_frames"] = n_frames
            cum += lead + n_frames
            # 该轮回复决策点相对语音结束的滞后
            delays.append((chunk_size - cum % chunk_size) % chunk_size * 40 or chunk_size * 40)

        # 尾部静音: 与上游同一套取整逻辑, 使总帧数落在 chunk 边界
        tail = rec["tail_silence_frames"] - (cum + rec["tail_silence_frames"]) % chunk_size
        if tail < 0:
            tail = (chunk_size - (cum % chunk_size)) % chunk_size
        segments.append(_gap_noise(rec["tail_noise_path"], rec["tail_noise_start_s"],
                                   tail * SAMPLES_PER_FRAME))
        rec["tail_silence_frames_actual"] = tail

        y = np.concatenate(segments)

        # ---- v4 噪声床叠加(整条, 含语音段): 事件轨 + 环境轨(再低 BED_AMBIENT_EXTRA_DB) ----
        if use_bed:
            snr = float(nprng.uniform(*_W["bed_snr"]))
            fade_n = _W["fade_n"]
            event = _tile_crossfade(rec["tail_noise_path"],
                                    rec["tail_noise_start_s"], len(y), fade_n)
            ambient = _tile_crossfade(rec["turns"][0]["leading_noise_path"],
                                      rec["turns"][0]["leading_noise_start_s"],
                                      len(y), fade_n)
            for track, extra in ((event, 0.0), (ambient, _W["bed_ambient_extra"])):
                t_db = _rms_db(track)
                if np.isfinite(t_db) and t_db > -100:
                    y = y + track * (10.0 ** ((speech_db - snr - extra - t_db) / 20.0))
            peak = float(np.abs(y).max())
            if peak > 0.99:
                y = y * (0.99 / peak)
            rec["gap_mode"], rec["bed_snr_db"] = "bed", round(snr, 1)
        else:
            rec["gap_mode"] = "v3"

        wav_path = os.path.join(_W["wavs_dir"], f"{rec['idx']}.wav")
        up._write_wav(wav_path, y)
        rec["concat_wav_path"] = wav_path
        return json.dumps(rec, ensure_ascii=False) + "\n", delays, None
    except Exception as e:
        return None, [], f"[step2zh idx {rec.get('idx')}] {type(e).__name__}: {e}"


def _step2_worker_init():
    """worker 初始化: 限制 BLAS 线程, 防 N 进程 × M 线程超订。"""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def step2_concat_audio_zh(input_jsonl, output_jsonl, wavs_dir, trimmed_dir, *,
                          chunk_size, align_mods, atten_db, trim_top_db,
                          gap_styles=GAP_STYLES, speech_aug_prob=SPEECH_AUG_PROB,
                          bed_prob=BED_PROB, seed=1337, workers=1):
    """上游 step2 的替代实现: [噪声→语音]×N→尾噪声, 带 trim/随机对齐/增益控制/fade。

    帧数计算完全复用上游 `_load_audio_aligned`(对修剪后的 wav 调用),
    与 step3 特征提取的卷积长度公式保持一致。
    workers>1 时多进程并行(soundfile 直读无子进程, 可安全并行;
    每条记录独立播种, 结果与并行度无关)。
    """
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)

    _W.update(dict(
        seed=seed, chunk_size=chunk_size, align_mods=align_mods, atten_db=atten_db,
        trim_top_db=trim_top_db,
        fade_n=int(CROSSFADE_MS / 1000 * 16000),
        wavs_dir=wavs_dir, trimmed_dir=trimmed_dir,
        gap_styles=gap_styles, floor_db=GAP_FLOOR_DB_RANGE, speech_aug_prob=speech_aug_prob,
        bed_prob=bed_prob, bed_snr=BED_SNR_DB_RANGE, bed_ambient_extra=BED_AMBIENT_EXTRA_DB,
    ))

    from tqdm import tqdm
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    delays_ms, n_err = [], 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        if workers <= 1:
            it = (_step2_one_record(l) for l in lines)
            for out_line, delays, err in tqdm(it, total=len(lines), desc="step2(zh)"):
                if err:
                    n_err += 1
                    print(err)
                else:
                    fout.write(out_line)
                    delays_ms.extend(delays)
        else:
            import multiprocessing as mp
            with mp.Pool(processes=workers, initializer=_step2_worker_init) as pool:
                for out_line, delays, err in tqdm(
                        pool.imap(_step2_one_record, lines, chunksize=8),
                        total=len(lines), desc=f"step2(zh)×{workers}"):
                    if err:
                        n_err += 1
                        print(err)
                    else:
                        fout.write(out_line)
                        delays_ms.extend(delays)

    if n_err:
        print(f"[step2zh] {n_err} 条失败(已跳过, 见上方日志)")
    if delays_ms:
        print(f"[延迟] 回复决策点滞后于语音结束: mean={np.mean(delays_ms):.0f}ms "
              f"min={np.min(delays_ms):.0f}ms max={np.max(delays_ms):.0f}ms "
              f"(align_mods={align_mods})")
    _report_gap_contrast(output_jsonl)


def _report_gap_contrast(step2_jsonl, n_check=6):
    """构造质量守门: 分样式抽样报告"语音 vs 空档"的响度差。
    v3 空档样式: 差值应 ≥ ~15dB(否则复现"训后不开口"坍缩);
    bed 噪声床样式: 差值即实际 SNR, 应落在 BED_SNR_DB_RANGE(约 4~21dB)——
    这里的边界线索是"语音消失、床还在", 不要求空档安静。"""
    try:
        checked = {"v3": [], "bed": []}
        n_mode = {"v3": 0, "bed": 0}
        snrs = []
        with open(step2_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                mode = rec.get("gap_mode", "v3")
                n_mode[mode] += 1
                if mode == "bed" and "bed_snr_db" in rec:
                    snrs.append(rec["bed_snr_db"])
                if len(checked[mode]) >= n_check:
                    continue
                wav, _ = sf.read(rec["concat_wav_path"], dtype="float32")
                pos, sp_db, gp_db = 0, [], []
                for t in rec["turns"]:
                    nL = t["leading_silence_frames"] * SAMPLES_PER_FRAME
                    nA = t["audio_frames"] * SAMPLES_PER_FRAME
                    gap = wav[pos:pos + nL]
                    # v3 的数字零空档不计入响度对比(其对比无穷大, 会掩盖真噪声空档的回归)
                    if nL and float(np.abs(gap).max()) > 0:
                        gp_db.append(_rms_db(gap))
                    pos += nL
                    sp_db.append(_rms_db(wav[pos:pos + nA]))
                    pos += nA
                if gp_db:
                    checked[mode].append(float(np.mean(sp_db) - np.mean(gp_db)))
        total = n_mode["v3"] + n_mode["bed"]
        if n_mode["bed"]:
            print(f"[床] 噪声床样式 {n_mode['bed']}/{total} 条 "
                  f"({100 * n_mode['bed'] / total:.0f}%), 目标 SNR "
                  f"mean={np.mean(snrs):.1f}dB range=[{min(snrs):.0f},{max(snrs):.0f}]dB")
        if checked["v3"]:
            c = float(np.mean(checked["v3"]))
            flag = "✓" if c >= 15 else "⚠️ 过小, 模型可能学不到开口时机!"
            print(f"[响度] v3空档样式: 语音比空档高 {c:.1f} dB (抽样 {len(checked['v3'])} 条) {flag}")
        if checked["bed"]:
            c = float(np.mean(checked["bed"]))
            flag = "✓" if 3.0 <= c <= 22.0 else "⚠️ 偏离 BED_SNR_DB_RANGE, 检查床增益!"
            print(f"[响度] bed床样式: 语音比空档高 {c:.1f} dB (抽样 {len(checked['bed'])} 条, "
                  f"即实测 SNR) {flag}")
    except Exception as e:
        print(f"[响度] 抽样检查失败(不影响构造): {e}")


def _build_gap_noise_dir(noise_dir, work_dir, exclude_pattern):
    """为空档填充建一个剔除了前景人声类噪声(Babble/广播)的符号链接池。"""
    if not exclude_pattern:
        return noise_dir
    import re
    import shutil
    pat = re.compile(exclude_pattern, re.I)
    dst = os.path.join(work_dir, "noise_gap_pool")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    n_keep = n_skip = 0
    for root, _, files in os.walk(noise_dir):
        for fn in files:
            if not fn.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                continue
            if pat.search(fn):
                n_skip += 1
                continue
            os.symlink(os.path.abspath(os.path.join(root, fn)),
                       os.path.join(dst, f"{n_keep:05d}_{fn}"))
            n_keep += 1
    print(f"[noise] 空档噪声池: 保留 {n_keep} 个, 剔除人声类 {n_skip} 个 "
          f"(pattern={exclude_pattern!r})")
    if n_keep == 0:
        sys.exit("[noise] 剔除后噪声池为空, 请放宽 --gap-noise-exclude")
    return dst


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
    ap.add_argument("--align-mods", default=",".join(map(str, ALIGN_TARGET_MODS)),
                    help="对齐目标 mod 集合(逗号分隔, 每轮随机选一)。mod=m → 决策点在语音"
                         "结束后 (10-m)*40ms。默认 4,5,6,7 → 120~240ms")
    ap.add_argument("--noise-atten-db", default=f"{NOISE_ATTEN_DB_RANGE[0]},{NOISE_ATTEN_DB_RANGE[1]}",
                    help="空档噪声相对语音的衰减范围 dB(逗号分隔 lo,hi, 每段随机)。默认 18,30")
    ap.add_argument("--no-noise-atten", action="store_true",
                    help="关闭空档噪声增益控制(危险: 会复现'训后不开口'坍缩)")
    ap.add_argument("--gap-noise-exclude", default=GAP_NOISE_EXCLUDE_PATTERN,
                    help="空档噪声池按文件名剔除的正则(前景人声类)。空串=不剔除")
    ap.add_argument("--no-v3-aug", action="store_true",
                    help="关闭 v3 部署对齐增强(空档三态+语音信道模拟), 回到纯衰减噪声空档")
    ap.add_argument("--bed-prob", type=float, default=BED_PROB,
                    help="v4 连续噪声床样式的样本占比(0=关闭)。床=事件+环境双轨噪声平铺"
                         "整条音频(语音段也叠), SNR~U(5,20)dB, 治'带噪真人声不开口'")
    ap.add_argument("--trim-top-db", type=float, default=TRIM_TOP_DB)
    ap.add_argument("--min-noise-len", type=int, default=20)
    ap.add_argument("--max-noise-len", type=int, default=60)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-steps", default="", help="逗号分隔跳过的步骤, 如 1,2,3(断点续跑)")
    ap.add_argument("--workers", type=int, default=16,
                    help="step2 并行进程数(soundfile 直读后纯 CPU, 可放心并行; 1=串行)")
    ap.add_argument("--no-sf-audio", action="store_true",
                    help="回退 ffmpeg 版 whisper.load_audio(调试对照用)")
    args = ap.parse_args()

    _lazy_imports(use_sf_audio=not args.no_sf_audio)
    skip = {s.strip() for s in args.skip_steps.split(",") if s.strip()}

    wd = os.path.abspath(args.work_dir)
    os.makedirs(wd, exist_ok=True)
    s1, s2, s3 = (os.path.join(wd, f"step{i}.jsonl") for i in (1, 2, 3))
    wavs, trimmed, feats = (os.path.join(wd, d) for d in ("wavs", "trimmed", "features"))

    tokenizer_dir, _, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(args.checkpoint_dir)

    align_mods = None if args.no_align else tuple(
        int(x) for x in args.align_mods.split(",") if x.strip())
    atten_db = None if args.no_noise_atten else tuple(
        float(x) for x in args.noise_atten_db.split(","))
    gap_noise_dir = _build_gap_noise_dir(args.noise_dir, wd, args.gap_noise_exclude)

    if "1" not in skip:
        up.step1_sample_silence(
            args.input, s1, noise_dir=gap_noise_dir,
            min_noise_len=args.min_noise_len, max_noise_len=args.max_noise_len,
            chunk_size=args.chunk_size, seed=args.seed,
        )
    if "2" not in skip:
        step2_concat_audio_zh(
            s1, s2, wavs, trimmed,
            chunk_size=args.chunk_size, align_mods=align_mods, atten_db=atten_db,
            trim_top_db=args.trim_top_db,
            gap_styles=None if args.no_v3_aug else GAP_STYLES,
            speech_aug_prob=0.0 if args.no_v3_aug else SPEECH_AUG_PROB,
            bed_prob=args.bed_prob, seed=args.seed, workers=args.workers,
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
