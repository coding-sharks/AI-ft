#!/usr/bin/env python3
"""moss-zh 多轮文本数据 -> cons_online_data(_zh).py 的输入格式转换。

输入 (每行一条 JSON):
  {"index": N, "ok": true, "user_speaker": ..., "dialog": "moss-18",
   "turns": [{"turn_idx": i, "role": "user"/"assistant", "text": ..., "emotion"?: ...}, ...]}

输出 (每行一条 JSON):
  {"dialog": "moss-18",
   "conversation": [{"audio_path", "assistant", "emotion"}, ...]}

audio_path 推断规则(真实数据, 集群上):
  {AUDIO_ROOT}/{dialog}/{dialog}_turn{user_turn_idx}.wav
  dialog 取记录的 "dialog" 字段(如 "moss-18", 已含前缀); 缺失则回退 f"moss-{index}"。

替身模式(--stub-audio-glob, 本机走通流程用):
  忽略真实路径, 把本地 wav 预处理(16k 单声道、修剪、截到 --stub-max-seconds)后
  循环分配给每个 user 轮。用于没有真实音频的机器上冒烟测试。

其他:
  --replicate N   每条输入复制 N 份输出(冒烟测试用: 1 条数据 -> N 条, 下游噪声随机各不相同)
  --check-audio   校验每个 audio_path 是否真实存在(集群上跑真实数据时建议开)
"""
import argparse
import glob
import json
import os
import sys

AUDIO_ROOT = "/apdcephfs/private_giannishu/api_call/data_finetuning/moss-zh"
VALID_EMO = {"happy", "sad", "angry", "surprise", "normal", "urgent"}


def _dialog_id(rec):
    did = rec.get("dialog")
    if did:
        return did
    if "index" in rec:
        return f"moss-{rec['index']}"
    return None


def prepare_stubs(globs, stub_dir, max_seconds):
    """把本地 wav 预处理成 16k 单声道、<=max_seconds 的替身音频, 返回路径列表。"""
    import librosa
    import numpy as np
    import soundfile as sf

    paths = []
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    paths = [p for p in paths if p.lower().endswith(".wav")]
    if not paths:
        sys.exit(f"[stub] 替身音频 glob 没匹配到任何 wav: {globs}")

    os.makedirs(stub_dir, exist_ok=True)
    out = []
    for i, p in enumerate(paths):
        try:
            y, _ = librosa.load(p, sr=16000, mono=True)
            y, _ = librosa.effects.trim(y, top_db=40)
            max_n = int(max_seconds * 16000)
            if len(y) > max_n:          # 太长取中段, 保证有语音内容
                s = (len(y) - max_n) // 2
                y = y[s: s + max_n]
            if len(y) < int(0.5 * 16000):
                print(f"[stub] 跳过过短音频 {p}")
                continue
            q = os.path.join(stub_dir, f"stub{i:02d}.wav")
            sf.write(q, y, 16000)
            out.append(q)
        except Exception as e:
            print(f"[stub] 跳过 {p}: {e}")
    if not out:
        sys.exit("[stub] 没有可用的替身音频")
    print(f"[stub] 准备了 {len(out)} 个替身音频 -> {stub_dir}")
    return out


def convert_record(rec, audio_root, stubs, stub_counter):
    """一条多轮记录 -> (输出 dict, 告警列表)。stubs 非空时用替身路径。"""
    did = _dialog_id(rec)
    turns = rec.get("turns", [])
    conv, warns = [], []
    if did is None:
        return {"conversation": []}, ["记录缺少 dialog/index, 无法推断路径"]

    i, n = 0, len(turns)
    while i < n:
        u = turns[i]
        if u.get("role") != "user":
            warns.append(f"{did}: 跳过非 user 轮 (turn_idx={u.get('turn_idx')}, role={u.get('role')})")
            i += 1
            continue
        if i + 1 < n and turns[i + 1].get("role") == "assistant":
            a = turns[i + 1]
            emo = str(a.get("emotion") or "normal").lower()
            if emo not in VALID_EMO:
                warns.append(f"{did}: 非法情感 {a.get('emotion')!r} (turn_idx={a.get('turn_idx')}) -> normal")
                emo = "normal"
            if stubs:
                ap = stubs[stub_counter[0] % len(stubs)]
                stub_counter[0] += 1
            else:
                ap = f"{audio_root}/{did}/{did}_turn{u['turn_idx']}.wav"
            conv.append({"audio_path": ap, "assistant": a["text"], "emotion": emo})
            i += 2
        else:
            warns.append(f"{did}: user turn_idx={u.get('turn_idx')} 无对应 assistant 回复, 跳过该轮")
            i += 1
    return {"dialog": did, "conversation": conv}, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--audio-root", default=AUDIO_ROOT)
    ap.add_argument("--check-audio", action="store_true")
    ap.add_argument("--replicate", type=int, default=1,
                    help="每条输入复制 N 份输出(冒烟用)")
    ap.add_argument("--stub-audio-glob", action="append", default=[],
                    help="替身音频 glob, 可多次给; 给了就进入替身模式")
    ap.add_argument("--stub-dir", default=None, help="替身音频预处理输出目录")
    ap.add_argument("--stub-max-seconds", type=float, default=4.0)
    args = ap.parse_args()
    root = args.audio_root.rstrip("/")

    stubs = []
    if args.stub_audio_glob:
        if not args.stub_dir:
            sys.exit("--stub-audio-glob 需要同时给 --stub-dir")
        stubs = prepare_stubs(args.stub_audio_glob, args.stub_dir, args.stub_max_seconds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stub_counter = [0]
    n_dialog = n_pairs = n_missing = n_dropped = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[跳过第 {ln} 行] JSON 解析失败: {e}")
                continue
            out, warns = convert_record(rec, root, stubs, stub_counter)
            for w in warns:
                print("[warn]", w)
            if not out["conversation"]:
                n_dropped += 1
                continue
            if args.check_audio and not stubs:
                for c in out["conversation"]:
                    if not os.path.isfile(c["audio_path"]):
                        n_missing += 1
                        print(f"[缺音频] {c['audio_path']}")
            for _ in range(max(1, args.replicate)):
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_dialog += 1
                n_pairs += len(out["conversation"])

    print(f"\n完成: 写出 {n_dialog} 条对话(replicate×{args.replicate}), 共 {n_pairs} 个配对"
          + (f", 丢弃 {n_dropped} 条空对话" if n_dropped else "")
          + (f"; 缺失音频 {n_missing} 个" if args.check_audio and not stubs else ""))
    if n_missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
