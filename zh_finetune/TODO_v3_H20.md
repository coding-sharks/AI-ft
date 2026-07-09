# AI-ft 中文微调 v3 —— 同步与重训 TODO(H20 侧 Claude 执行清单)

> 本文件自包含。执行环境:H20 机器,仓库 `/apdcephfs/private_giannishu/api_call/AI-ft`,
> conda env `audiointeraction`(常驻 CephFS)。逐条执行,汇报清单在文末。

## 背景(2 句话)

旧模型(`runtime_rw_filtered_20260708` 下的产物)在训练分布音频上 P(TEXT_BEGIN)=0.9999、逐轮精准开口,但对浏览器麦克风音频全程沉默(<0.4%)→ 定性为 **OOD 不泛化**:训练音频世界(响亮 MS-SNSD 空档 + 干净 TTS)≠ 麦克风世界(noiseSuppression 数字零静音 + AGC/NS 染色的真人声)。远端已推送 v3 修复:**空档三态**(衰减真噪声 45% / 数字绝对零 30% / 极安白噪底 25%)+ **语音信道模拟**(增益/变速/轻混响/频谱倾斜,p=0.7)+ 安静空档增益控制 + 120~240ms 随机对齐 + epochs 2。

---

## Task 1 同步代码(1 分钟)

```bash
cd /apdcephfs/private_giannishu/api_call/AI-ft
git pull                      # 目标 HEAD = 45eb9d9
git log --oneline -4          # 应看到: 45eb9d9 / 166790e / 09d0c7e / 7731184
```

## Task 2 路径体检(5 分钟,重要)

`run_all.sh` 固定使用 `$REPO/zh_finetune/runtime` 与 `$REPO/checkpoints/audiointeraction`:

- [ ] 确认 `zh_finetune/runtime/` 存在且含 `init.pt` 和 `noise/`。
      旧产物若在 `runtime_rw_filtered_20260708/`,把 `init.pt`、`noise/` 挪回或软链到 `runtime/`。
- [ ] 确认 `checkpoints/audiointeraction/` 存在(model_config.yaml + 4 片 safetensors +
      audiointeraction_ChunkwisedEncoder.pth + qwen25OmniConfig/)。
      若真身在 `/apdcephfs/private_giannishu/models/checkpoints`,软链过来,避免重下 14GB:
      `ln -s /apdcephfs/private_giannishu/models/checkpoints checkpoints/audiointeraction`

## Task 3 诊断实验(30 分钟;与 Task 5 的数据构造并行,结果只影响预期、不阻塞重训)

用现有 `/tmp/diag_probs.py` 逐帧测 P(TEXT_BEGIN),两组:

**(a) 实验 II —— 麦克风语音包进"旧训练风格"空档,喂旧模型 best_step000400**

```python
# /tmp/make_wrapped.py
import numpy as np, soundfile as sf, glob, random
mic, _ = sf.read("/apdcephfs/private_giannishu/Audio-Interaction/zh_demo/recordings/"
                 "20260709_042650_301/_merged.wav", dtype="float32")
speech = mic[16*6400:23*6400]          # 语音段2(帧16-22, "介绍手机"那句)
pool = glob.glob("zh_finetune/runtime/noise/MS-SNSD/noise_train/*.wav")
def gap(sec, db):
    n, _ = sf.read(random.choice(pool), dtype="float32"); n = n[:int(sec*16000)]
    return n * (np.sqrt((speech**2).mean()) * 10**(-db/20) / (np.sqrt((n**2).mean())+1e-9))
for tag, db in [("loud3db", 3.0), ("quiet24db", 24.0)]:
    sf.write(f"/tmp/mic_wrapped_{tag}.wav",
             np.concatenate([gap(1.6, db), speech, gap(1.2, db)]), 16000)
```

判读:
- `loud3db` 版开口 → **空档纹理是旧模型的死因**(v3 直接治,预期乐观);
- 两版都沉默 → **信道/真人声轴主导**(v3 的信道模拟 + Task 4 锚定集就是为此准备,锚定集变为必做)。

**(b) 对照 —— 底座 `zh_finetune/runtime/init.pt` 直接喂原始麦克风 `_merged.wav`**

若底座有明显开口倾向(哪怕英文回复)→ 坐实"微调收窄分布"。

→ 记录 (a)(b) 的逐帧 P 表,写入汇报。

## Task 4 真人语音数据(两级;**这是本轮方案的核心**)

> 依据链:①实验 II 两版包装全沉默 → 语音本身(真人声/信道)是 OOD 主轴;
> ②底座对麦克风 P=0.94/0.99 能开口、微调后 <0.4% → LM 全参微调把底座本有的
> 真人声判读能力训漂了(audio tower 冻结只保"耳朵"不保"判读",特征同、判决异);
> ③论文 Table 9:原版 StreamAudio-2M 真人语音 ≈63 万条(CommonVoice/GigaSpeech/
> LibriSpeech/CoVoST2/**AISHELL 141k**…)与 TTS ≈54 万条**对半开**——我们的 9k 纯
> CosyVoice2 是这个模型第一次吃"全合成餐"。本 Task = 把真人那一半按比例放回去。

### 4A 真实麦克风锚定集(必做,半天,需人类用户)
- [ ] 用 zh_demo 实录 **100~300 条**真人口语问句(信道与部署 100% 一致)。
- [ ] 音频按约定放置:`{AUDIO_ROOT}/mic-0001/mic-0001_turn1.wav` …
      (AUDIO_ROOT = `/apdcephfs/private_giannishu/api_call/data_finetuning/moss-zh`)
- [ ] 每条配 LLM 写的中文回复,data.jsonl 追加:
      `{"dialog":"mic-0001","turns":[{user...},{assistant(含emotion)...}]}`
      **同一条记录连写 3~5 行**(=上采样)。
- [ ] 另留 **10~20 条实录不进 data.jsonl** —— held-out 验收集(Task 6 用)。

### 4B 开源真人语音混入 —— ✅ 已由 A100 侧代工完成,直接取用(无需下载 AISHELL 原始包!)

成品数据仓:**https://github.com/coding-sharks/AI-ft-data**(公开,匿名可 clone)
内容:AISHELL-1 抽样 **12,000 句 / 2,681 条多轮对话 / 覆盖全部 400 个说话人**,
Qwen2.5-7B-Instruct 生成回复(0 兜底 / 99% 唯一 / 0 违禁词),16kHz FLAC 0.81GB;
已用 convert_to_online_input.py 实测 12,000 配对 0 缺失(依赖其 .wav→.flac 回退,
AI-ft 需 ≥ ea8dea8)。

H20 侧三条命令:
```bash
git clone https://github.com/coding-sharks/AI-ft-data.git /tmp/AI-ft-data
cp -r /tmp/AI-ft-data/audio/aishell-* \
      /apdcephfs/private_giannishu/api_call/data_finetuning/moss-zh/
cat /tmp/AI-ft-data/aishell_mix.jsonl >> /path/to/合并后的data.jsonl
```
- [ ] 执行上面三条(与 4A 合并到同一个 data.jsonl)
- [ ] **⚠️ 铁律不变:真人语音轮全部已配回复;后续自行加数据时也绝不许标
      `<no need to response>`**(会重新教出"真人声→沉默")。

### 4C 配比核算(写进汇报)
合并后统计:真人对话条数 /(TTS 9k + 真人)应落在 **25%~40%**;
不足则加大 4A 上采样倍数或 4B 抽样量。

## Task 5 重建数据 + 重训(v3)

```bash
# ⚠️ 必须删旧产物: config 的 resume:"auto" 会捡起旧 checkpoint 续训, 不删等于白修
rm -rf zh_finetune/runtime/train_output/zh-full
rm -rf zh_finetune/runtime/data_full/work zh_finetune/runtime/data_full/train_jsonl
rm -f  zh_finetune/runtime/data_full/online_input.jsonl   # 若 data.jsonl 已并入锚定集

tmux new -s zhft
bash zh_finetune/run_all.sh --full --devices 1 \
     --data /path/合并后的data.jsonl --workers 64
```

起跑核对守门日志(缺一即停下排查):

```
[noise] 空档噪声池: 保留 … 剔除人声类 …
[延迟] 回复决策点滞后于语音结束: mean=180ms min=120ms max=240ms (align_mods=(4, 5, 6, 7))
[响度] 语音比空档噪声高 ≥15dB ✓
[flash] ✅   [ckpt] ✅
显存 ~73-78G / 97.8G;iter ~2.5-3.5s
```

训练量:9k TTS + 4A/4B 真人数据 ≈ 13-15k 条对话 → 2 epoch ≈ 400-470 优化步,
单卡 H20 约 **10-13 小时**;中断后重跑同命令自动续训。
(可选:磁盘充裕的话把 config_zh_full.yaml 的 save_interval 100→50,
 给 Task 6.0 的 checkpoint 扫描更细的粒度;每个存档 ~40G,自行权衡。)

## Task 6 验收(⚠️ 协议已变 —— 用真实录音验收 + 按真实指标选 checkpoint!)

**6.0 checkpoint 扫描(关键保险,Task 3(b) 的直接推论)**:分布收窄随训练步数渐进
加深——底座(step 0)对麦克风 P=0.94/0.99,旧模型训到后期 <0.4%。因此**不要默认部署
final**:对 `train_output/zh-full/` 下所有存档(step-100/step-200/…/best-*/final)
用 held-out 实录逐个测"语音结束点 P(TEXT_BEGIN)",**选真实麦克风表现最好的那个部署**
(用现有 diag 脚本对 ckpt 路径循环即可)。若最优点明显早于 final,说明还在收窄,
下轮加大锚定集占比或减 epoch。

1. **held-out 实录** → 逐帧 P(TEXT_BEGIN):语音结束点应 argmax=TEXT_BEGIN(P 至少几十%),
   静音段/说话中段保持 KEEP_SILENCE;
2. 训练分布 wav(`data_full/work/wavs/0.wav`)回归检查:每轮各开口一次;
3. zh_demo 端到端实测:`server_zh.py` 换 **6.0 选出的 checkpoint**,
   prompt 保持与训练一致的 `ZH_SYSTEM_PROMPT`(zh_config.py 单一来源)。

## 汇报清单(执行完发回)

- [x] Task 3 (a)(b) 的逐帧 P 表与判读结论
      (2026-07-09 已完成:两版包装全沉默→信道/真人声轴主导;底座对麦克风 2/36 开口
       P=0.94/0.99→坐实微调收窄。详见 zh_demo/DIAGNOSIS.md §0.5)
- [ ] Task 5 四条守门日志 + step 50 首次 eval 的 `val_moss_zh`
- [ ] Task 6.0 checkpoint 扫描表(各存档在 held-out 实录上的 P)+ 选定的部署 ckpt
- [ ] Task 6 三项验收结果(重点:held-out 实录上的开口表现)
