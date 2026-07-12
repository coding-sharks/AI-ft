#!/usr/bin/env python3
"""RAMC 提问切片包: 把 MagicData-RAMC(180h 真人自由对话)切成"一句提问 = 一条单轮对话"。

与 aishell-mix 的分工不同: 本脚本只产出【用户轮音频 + 转写】, 回复由 H20 侧用
DeepSeek API 生成(见数据仓的 gen_replies_deepseek.py)——所以没有 replies 阶段。

产出(与 moss/aishell 数据同构, 单轮):
  <out>/audio/ramc-G00000096-0001/ramc-G00000096-0001_turn1.flac   (16k 单声道)
  <out>/ramc_selection.jsonl   每行 {"index","dialog","speaker","gender","src","t0","t1",
                                     "turns":[{"turn_idx":1,"role":"user","text":...}]}
  (assistant 轮由 gen_replies_deepseek.py 补齐后另存 ramc_mix.jsonl)

选取逻辑:
  - 只要疑问句(--only-questions, 默认开): 含 ?/? 或 疑问词(什么/怎么/为什么/哪/多少/
    几/谁/干嘛/如何/咋)或 句尾"吗/呢"——这是本包的使命(自发口语+真实提问)
  - 质量过滤: 时长 1.0~9.0s; 清洗后 5~40 字; 剔除含标注记号([+]重叠/[*]噪声/
    [LAUGHTER]/[SONANT])的行; 剔除纯语气词 backchannel
  - 重叠过滤: 句子时间区间与【对方说话人】区间重叠 > 0.3s 的丢弃(双人混单声道)
  - 说话人轮转采样: 每轮每人 1 句, 铺满全部说话人(沿用 aishell-mix 的教训)
  - 切片带边缘余量 0.1s(不侵入相邻句), 下游构造还会 trim, 这里只保证不切掉音素

用法(在仓库根; src 目录下应有解包出的 train/ dev/ test, 各含 wav/ 与 txt/):
  python zh_finetune/tools/build_ramc_mix.py \
      --src /mnt/data/guoqiang/SDMs/ramc_work/extracted \
      --out /mnt/data/guoqiang/SDMs/AI-ft-data --n-utts 10000
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

MARKER = re.compile(r"\[[^\]]*\]")                      # [+] [*] [LAUGHTER] [SONANT]
BACKCHANNEL = re.compile(r"[嗯啊哦噢呃对是的呀哈嘿诶欸嗨行好吧了没错就,，。!！?？\s]*$")
# 两级疑问判定: 强 = 问号 / 句尾"吗|呢" —— 几乎必是提问;
#               弱 = 句中疑问词 —— 有误报("不知道该干嘛"是陈述), 只作强句不足时的补充
Q_STRONG = re.compile(r"[?？]|(?:吗|呢)[,，。!！\s]*$")
Q_WEAK = re.compile(
    r"什么|怎么|怎样|为什么|为啥|凭什么|哪|多少|谁|干嘛|干啥|如何|咋|几点|几个|几天|几年|多久|多大")
BLOCKLIST = re.compile(r"操|草泥|尼玛|你妈|他妈|妈的|卧槽|我靠|傻逼|傻叉|智障|贱|滚蛋|卵|屌|艹|去死")
LINE = re.compile(r"^\[([\d.]+),([\d.]+)\]\t(\S+)\t([^\t]*)\t(.*)$")


def parse_conversation(txt_path):
    """→ [(t0, t1, spk, gender, text_raw)], 按时间排序。"""
    utts = []
    for line in open(txt_path, encoding="utf-8"):
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        t0, t1, spk, meta, text = m.groups()
        gender = meta.split(",")[0] if "," in meta else ""
        utts.append((float(t0), float(t1), spk, gender, text.strip()))
    utts.sort(key=lambda u: u[0])
    return utts


def overlap_with_other(utt, all_utts, limit=0.3):
    """该句与其他说话人句子的时间重叠总量是否超限。"""
    t0, t1, spk = utt[0], utt[1], utt[2]
    total = 0.0
    for o0, o1, ospk, _, _ in all_utts:
        if ospk == spk or o1 <= t0:
            continue
        if o0 >= t1:
            break
        total += min(t1, o1) - max(t0, o0)
        if total > limit:
            return True
    return False


def collect_candidates(src_root, min_dur, max_dur, min_chars, max_chars, only_q):
    """扫描全部对话 → speaker -> [cand], cand=(wav_path, t0, t1, gender, text_clean)。"""
    by_spk = defaultdict(list)
    stats = defaultdict(int)
    txts = sorted(src_root.rglob("txt/*.txt"))
    print(f"[cut] 发现 {len(txts)} 场对话转写")
    for txt in txts:
        wav = txt.parent.parent / "wav" / (txt.stem + ".wav")
        if not wav.is_file():
            stats["no_wav"] += 1
            continue
        utts = parse_conversation(txt)
        for u in utts:
            t0, t1, spk, gender, raw = u
            stats["total"] += 1
            if MARKER.search(raw):                 # 重叠/噪声/笑声标注 → 不干净, 弃
                stats["marker"] += 1
                continue
            text = raw.strip()
            if not (min_dur <= t1 - t0 <= max_dur):
                stats["dur"] += 1
                continue
            if not (min_chars <= len(text) <= max_chars):
                stats["len"] += 1
                continue
            if BACKCHANNEL.fullmatch(text):
                stats["backchannel"] += 1
                continue
            if BLOCKLIST.search(text):
                stats["blocked"] += 1
                continue
            if Q_STRONG.search(text):
                tier = 0
            elif Q_WEAK.search(text):
                tier = 1
            else:
                tier = 2
            if only_q and tier == 2:
                stats["not_question"] += 1
                continue
            if overlap_with_other(u, utts):
                stats["overlap"] += 1
                continue
            by_spk[spk].append((tier, str(wav), t0, t1, gender, text))
            stats["kept"] += 1
            stats[f"tier{tier}"] += 1
    print("[cut] 过滤统计:", dict(stats))
    return by_spk


def cut_and_write(by_spk, out_dir, n_utts, seed):
    rng = random.Random(seed)
    speakers = sorted(by_spk.keys())
    rng.shuffle(speakers)
    for spk in speakers:
        rng.shuffle(by_spk[spk])
        by_spk[spk].sort(key=lambda c: -c[0])   # 强疑问句(tier0)放尾部 → pop 先取

    # 说话人轮转: 每轮每人出 1 句, 直到抽满(每人内部强疑问句优先)
    picked = []
    seen_text = set()                      # 全局文本去重(口头禅重复句很多)
    while len(picked) < n_utts:
        progressed = False
        for spk in speakers:
            if len(picked) >= n_utts:
                break
            pool = by_spk[spk]
            while pool:
                tier, wav_path, t0, t1, gender, text = pool.pop()
                if text in seen_text:
                    continue
                seen_text.add(text)
                picked.append((spk, wav_path, t0, t1, gender, text, tier))
                progressed = True
                break
        if not progressed:
            break
    n_spk = len({p[0] for p in picked})
    n_strong = sum(1 for p in picked if p[6] == 0)
    print(f"[cut] 选中 {len(picked)} 句 / 覆盖说话人 {n_spk} / 强疑问句 {n_strong} "
          f"({100 * n_strong / max(len(picked), 1):.0f}%)")

    audio_root = out_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    picked.sort(key=lambda p: p[1])        # 按源 wav 排序 → 单文件缓存生效
    rows, seq_by_spk = [], defaultdict(int)
    wav_cache = {}
    for i, (spk, wav_path, t0, t1, gender, text, tier) in enumerate(picked):
        if wav_path not in wav_cache:
            wav_cache.clear()              # 只缓存当前文件(picked 未按文件排序时避免爆内存)
            data, sr = sf.read(wav_path, dtype="float32")
            assert sr == 16000, f"非16k: {wav_path}"
            wav_cache[wav_path] = data
        data = wav_cache[wav_path]
        margin = 0.1
        s = max(0, int((t0 - margin) * 16000))
        e = min(len(data), int((t1 + margin) * 16000))
        clip = data[s:e]
        if len(clip) < 8000 or float(np.sqrt((clip ** 2).mean())) < 1e-4:
            continue                        # 空音频守门
        seq_by_spk[spk] += 1
        dialog = f"ramc-{spk}-{seq_by_spk[spk]:04d}"
        d = audio_root / dialog
        d.mkdir(exist_ok=True)
        sf.write(d / f"{dialog}_turn1.flac", clip, 16000)
        rows.append({
            "index": len(rows), "dialog": dialog, "speaker": spk, "gender": gender,
            "src": Path(wav_path).stem, "t0": round(t0, 3), "t1": round(t1, 3),
            "q_tier": tier,  # 0=强疑问(问号/句尾吗呢) 1=含疑问词
            "turns": [{"turn_idx": 1, "role": "user", "text": text}],
        })
        if (i + 1) % 1000 == 0:
            print(f"[cut] 切片 {i + 1}/{len(picked)}")

    sel = out_dir / "ramc_selection.jsonl"
    with open(sel, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[cut] 完成: {len(rows)} 条单轮 → {sel}")
    return rows


def verify(out_dir, rows, n_check=32):
    """抽查切片可解码、16k、非静音; 与 jsonl 配对。"""
    rng = random.Random(7)
    bad = 0
    for r in rng.sample(rows, min(n_check, len(rows))):
        p = out_dir / "audio" / r["dialog"] / f"{r['dialog']}_turn1.flac"
        try:
            d, sr = sf.read(p, dtype="float32")
            assert sr == 16000 and len(d) > 8000 and float(np.abs(d).max()) > 1e-3
        except Exception as e:
            print(f"[verify] ✗ {p}: {e}")
            bad += 1
    print(f"[verify] 抽查 {min(n_check, len(rows))} 条, 失败 {bad}")
    missing = sum(1 for r in rows
                  if not (out_dir / "audio" / r["dialog"] / f"{r['dialog']}_turn1.flac").is_file())
    print(f"[verify] jsonl↔音频配对缺失: {missing}")
    return bad == 0 and missing == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="解包根目录(下含 train/dev/test, 各有 wav/ txt/)")
    ap.add_argument("--out", required=True, help="数据包输出目录(AI-ft-data 仓)")
    ap.add_argument("--n-utts", type=int, default=10000)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=9.0)
    ap.add_argument("--min-chars", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=40)
    ap.add_argument("--all-styles", action="store_true", help="不限疑问句(默认只要疑问句)")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    by_spk = collect_candidates(src, a.min_dur, a.max_dur, a.min_chars, a.max_chars,
                                only_q=not a.all_styles)
    if not by_spk:
        sys.exit("没有候选句, 检查 --src 结构(应含 */txt/*.txt 与 */wav/*.wav)")
    rows = cut_and_write(by_spk, out, a.n_utts, a.seed)
    ok = verify(out, rows)
    print("✅ 全部通过" if ok else "❌ 有失败项, 检查上方日志")


if __name__ == "__main__":
    main()
