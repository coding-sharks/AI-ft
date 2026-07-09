#!/usr/bin/env python3
"""TODO Task 4B: 把 AISHELL-1 真人语音加工成可直接混入训练的数据包。

产出与 moss 数据完全同构:
  <out>/audio/aishell-S0002-001/aishell-S0002-001_turn1.wav   (用户轮音频, 16k wav)
  <out>/aishell_mix.jsonl   每行 {"index","dialog","turns":[{turn_idx,role,text[,emotion]}...]}

三阶段(--stage all 顺序执行, 可单独重跑):
  select   解析 transcript → 过滤 → 按 speaker 分组成 3~6 句/对话 → 抽取所需 speaker 的
           内层 tar.gz → selection.jsonl
  replies  本地 Qwen2.5-7B-Instruct 批量为每句转写生成简短口语回复 + 情感标签(6类白名单)
           ⚠️ 铁律: 真人语音轮一律给回复, 绝不生成"无需回应"(会重新教出"真人声→沉默")
  package  复制/重命名音频到目标布局 + 写 jsonl + 统计

用法(在仓库根):
  python zh_finetune/tools/build_aishell_mix.py --stage all \
      --aishell-tgz zh_finetune/runtime/aishell_src/data_aishell.tgz \
      --qwen zh_finetune/runtime/tools_models/qwen2.5-7b-instruct \
      --out zh_finetune/runtime/aishell_mix --n-utts 12000
完成后用 convert_to_online_input.py --check-audio 验证格式(见 --help 尾注)。
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

VALID_EMO = {"happy", "sad", "angry", "surprise", "normal", "urgent"}
SYS_PROMPT = (
    "你是一个中文语音助手。用户对你说了一句话(可能是陈述句、新闻播报腔或提问),"
    "你要给出简短、自然、口语化的中文回应(10~40字):陈述句就自然接话/简评/补充,"
    "提问就直接回答。绝对不要回复'无需回应'或表示不理睬。"
    "输出恰好两行:\n回复:<你的回应>\n情感:<happy/sad/angry/surprise/normal/urgent 之一, 拿不准写 normal>"
)
FALLBACK_REPLIES = [
    "嗯,你说的这个我明白了,还有什么想聊的吗?",
    "原来是这样,听起来确实值得关注。",
    "好的,这个信息我记下了,你可以接着说。",
    "明白,这事儿确实挺有意思的。",
]


def _parse_reply(gen):
    """三级容错解析 → (reply|None, emotion)。
    ①两行协议 "回复:… / 情感:…";②严格 JSON {"reply":…};
    ③观测到的混合体: 纯文本回复 + 尾随 {"emotion":…}(或裸文本)。"""
    emo = "normal"
    m = re.search(r"情感\s*[::]\s*([a-zA-Z]+)", gen)
    if not m:
        m = re.search(r'"emotion"\s*:\s*"([a-zA-Z]+)"', gen)
    if m and m.group(1).lower() in VALID_EMO:
        emo = m.group(1).lower()

    r = re.search(r"回复\s*[::]\s*(.+)", gen)
    if r:
        reply = r.group(1).strip()
    else:
        j = re.search(r'\{[^{}]*"reply"\s*:\s*"([^"]+)"[^{}]*\}', gen, re.S)
        if j:
            reply = j.group(1).strip()
        else:
            # 混合体/裸文本: 去掉尾随 JSON 片段与代码栅栏后取正文
            body = re.sub(r"\{[^{}]*\}\s*$", "", gen).strip()
            body = body.strip("`").strip()
            reply = body.splitlines()[0].strip() if body else ""
    reply = reply.strip('"“”').strip()
    if (not reply or len(reply) < 4 or len(reply) > 80
            or "无需回应" in reply or "不回应" in reply):
        return None, emo
    return reply[:60], emo


# ---------------- stage: select ----------------

def stage_select(aishell_tgz, workdir, n_utts, min_chars, max_chars,
                 turns_per_dialog, seed):
    rng = random.Random(seed)
    workdir.mkdir(parents=True, exist_ok=True)
    src_root = workdir / "extracted"

    # 1) 外层 tgz: 先只解出 transcript(小文件), wav 的内层 tar.gz 按需解
    trans_path = None
    with tarfile.open(aishell_tgz, "r:gz") as tf:
        for m in tf:
            if m.name.endswith("aishell_transcript_v0.8.txt"):
                tf.extract(m, workdir)
                trans_path = workdir / m.name
                break
    if trans_path is None:
        sys.exit("transcript 未在 tgz 中找到")
    print(f"[select] transcript: {trans_path}")

    # 2) 解析转写: "BAC009S0002W0122 而 对 楼市 ..." → utt_id, 无空格文本
    utts = defaultdict(list)  # speaker -> [(utt_id, text)]
    for line in open(trans_path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        uid, text = parts[0], "".join(parts[1:])
        if not (min_chars <= len(text) <= max_chars):
            continue
        spk = re.match(r"BAC009(S\d{4})W\d+", uid)
        if spk:
            utts[spk.group(1)].append((uid, text))
    print(f"[select] 过滤后: {sum(len(v) for v in utts.values())} 句 / {len(utts)} 说话人")

    # 3) 按说话人组对话(同一对话内同一说话人), 轮数 3~6 随机, 直到抽满 n_utts
    speakers = sorted(utts.keys())
    rng.shuffle(speakers)
    dialogs, total = [], 0
    for spk in speakers:
        pool = utts[spk][:]
        rng.shuffle(pool)
        i = 0
        while i + turns_per_dialog[0] <= len(pool) and total < n_utts:
            k = rng.randint(*turns_per_dialog)
            chunk = pool[i:i + k]
            i += k
            if len(chunk) < turns_per_dialog[0]:
                break
            dialogs.append({"speaker": spk, "utts": chunk})
            total += len(chunk)
        if total >= n_utts:
            break
    print(f"[select] 组成对话 {len(dialogs)} 条 / 共 {total} 句")

    # 4) 只解压被选中说话人的内层 tar.gz
    need = sorted({d["speaker"] for d in dialogs})
    with tarfile.open(aishell_tgz, "r:gz") as tf:
        members = {os.path.basename(m.name): m for m in tf
                   if m.name.endswith(".tar.gz")}
        for j, spk in enumerate(need):
            inner_name = f"{spk}.tar.gz"
            if inner_name not in members:
                print(f"[select][warn] 缺内层包 {inner_name}")
                continue
            tf.extract(members[inner_name], workdir / "inner")
            inner_path = next((workdir / "inner").rglob(inner_name))
            with tarfile.open(inner_path, "r:gz") as itf:
                itf.extractall(src_root)
            inner_path.unlink()
            if (j + 1) % 20 == 0:
                print(f"[select] 解压说话人 {j+1}/{len(need)}")
    # 建 utt_id -> wav 路径索引
    wav_index = {p.stem: str(p) for p in src_root.rglob("*.wav")}
    print(f"[select] 解压 wav {len(wav_index)} 个")

    kept = []
    for d in dialogs:
        paths = [(uid, txt, wav_index.get(uid)) for uid, txt in d["utts"]]
        paths = [(u, t, p) for u, t, p in paths
                 if p and os.path.getsize(p) > 50_000]  # >~1.5s
        if len(paths) >= turns_per_dialog[0]:
            kept.append({"speaker": d["speaker"], "utts": paths})
    with open(workdir / "selection.jsonl", "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n = sum(len(d["utts"]) for d in kept)
    print(f"[select] 完成: {len(kept)} 对话 / {n} 句 → selection.jsonl")


# ---------------- stage: replies ----------------

def stage_replies(workdir, qwen_dir, batch_size, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dialogs = [json.loads(l) for l in open(workdir / "selection.jsonl", encoding="utf-8")]
    texts = [t for d in dialogs for (_, t, _) in d["utts"]]
    print(f"[replies] 需生成 {len(texts)} 条回复; 加载 {qwen_dir} ...")

    tok = AutoTokenizer.from_pretrained(qwen_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        qwen_dir, dtype=torch.bfloat16, device_map=device)  # 需 accelerate(requirements 已含)
    model.eval()

    replies = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="replies"):
        batch = texts[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS_PROMPT},
             {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=96, do_sample=True,
                                 temperature=0.8, top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, seq in enumerate(out):
            gen = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            reply, emo = _parse_reply(gen)
            if reply is None:
                reply, emo = random.choice(FALLBACK_REPLIES), "normal"
            replies.append({"reply": reply, "emotion": emo})

    with open(workdir / "replies.jsonl", "w", encoding="utf-8") as f:
        for r in replies:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_fb = sum(1 for r in replies if r["reply"] in FALLBACK_REPLIES)
    print(f"[replies] 完成 {len(replies)} 条(兜底占 {n_fb}, {100*n_fb/len(replies):.1f}%)")


# ---------------- stage: package ----------------

def stage_package(workdir, out_dir):
    dialogs = [json.loads(l) for l in open(workdir / "selection.jsonl", encoding="utf-8")]
    replies = [json.loads(l) for l in open(workdir / "replies.jsonl", encoding="utf-8")]
    audio_root = out_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    import soundfile as _sf

    it = iter(replies)
    n_dialog = n_utt = 0
    per_spk = defaultdict(int)
    with open(out_dir / "aishell_mix.jsonl", "w", encoding="utf-8") as fout:
        for d in dialogs:
            spk = d["speaker"]
            per_spk[spk] += 1
            did = f"aishell-{spk}-{per_spk[spk]:03d}"
            ddir = audio_root / did
            ddir.mkdir(exist_ok=True)
            turns = []
            for k, (uid, text, wav) in enumerate(d["utts"]):
                r = next(it)
                u_idx = 2 * k + 1
                # 无损 FLAC 交付(体积≈wav 一半, GitHub Release 2GB 限额友好;
                # 转换脚本已支持 .wav→.flac 回退, 训练管线经 soundfile 按文件头识别格式)
                y, sr = _sf.read(wav, dtype="int16")
                _sf.write(ddir / f"{did}_turn{u_idx}.flac", y, sr)
                turns.append({"turn_idx": u_idx, "role": "user", "text": text})
                turns.append({"turn_idx": u_idx + 1, "role": "assistant",
                              "text": r["reply"], "emotion": r["emotion"]})
                n_utt += 1
            fout.write(json.dumps(
                {"index": n_dialog, "dialog": did, "ok": True, "turns": turns},
                ensure_ascii=False) + "\n")
            n_dialog += 1
    size_gb = sum(f.stat().st_size for f in audio_root.rglob("*.flac")) / 1e9
    print(f"[package] 完成: {n_dialog} 对话 / {n_utt} 句 / 音频 {size_gb:.2f} GB → {out_dir}")
    print(f"[package] 交付物: {out_dir}/aishell_mix.jsonl + {out_dir}/audio/aishell-*/")
    print("[package] H20 侧用法: 把 audio/ 下所有 aishell-* 目录放进 AUDIO_ROOT(moss-zh)/,"
          " 把 aishell_mix.jsonl 的行追加进 data.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["select", "replies", "package", "all"])
    ap.add_argument("--aishell-tgz", type=Path, required=True)
    ap.add_argument("--qwen", type=Path, help="Qwen2.5-7B-Instruct 本地目录(replies 阶段需要)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-utts", type=int, default=12000)
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=30)
    ap.add_argument("--turns-min", type=int, default=3)
    ap.add_argument("--turns-max", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    workdir = args.out / "work"
    if args.stage in ("select", "all"):
        stage_select(args.aishell_tgz, workdir, args.n_utts,
                     args.min_chars, args.max_chars,
                     (args.turns_min, args.turns_max), args.seed)
    if args.stage in ("replies", "all"):
        assert args.qwen, "--qwen 必填(replies 阶段)"
        stage_replies(workdir, args.qwen, args.batch_size, args.device)
    if args.stage in ("package", "all"):
        stage_package(workdir, args.out)


if __name__ == "__main__":
    main()
