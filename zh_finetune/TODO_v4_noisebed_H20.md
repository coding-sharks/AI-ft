# AI-ft 中文微调 v4 —— 噪声床确认实验与重训 TODO(H20 侧 Claude 执行清单)

> 本文件自包含。执行环境:H20 机器,仓库 `/apdcephfs/private_giannishu/api_call/AI-ft`,
> conda env `audiointeraction`。逐条执行,汇报清单在文末。
> 前置:v3 数据(TTS 9k + aishell 12k + ramc 10k)已训完,开口情况持续变好但仍经常不开口。

## 背景(3 句话)

v1-v3 的训练数据里,**语音段自始至终是干净的**(噪声只填在空档里,且空档被压到语音
−18~30dB)。原论文(StreamAudio-2M, Algorithm 4)的做法是**双轨连续噪声床平铺整条音频
——语音段上也叠噪声**,事件轨 SNR~U(5,20)dB + 环境轨再低 5dB;真实麦克风的"人声+环境噪声"
混合纹理因此在我们的训练分布之外,且我们模型的开口线索依赖"空档极安静",环境有持续噪声时
该线索永不出现。v4 = 把噪声床加进数据构造(已实现并在 A100 冒烟验证),但**重训前先花
30 分钟做确认实验**,证实"语音带噪→不开口"这条因果链,避免白跑 10 小时。

## Task 1 同步代码(1 分钟)

```bash
cd /apdcephfs/private_giannishu/api_call/AI-ft
git pull            # 应看到 v4 相关 commit(cons_online_data_zh.py 噪声床 + 本文件)
```

## Task 2 SNR 退化确认实验(30 分钟,决定要不要重训)

**原理**:拿一条模型【已知会开口】的训练分布 wav,人工按 SNR 20/15/10/5 dB 叠噪声床,
测语音结束点 P(TEXT_BEGIN) 随 SNR 的退化曲线。

```bash
# 1) 生成实验音频(无需 GPU)
python zh_finetune/tools/make_snr_sweep.py \
    --src zh_finetune/runtime/data_full/work/wavs/0.wav \
    --noise-dir zh_finetune/runtime/noise \
    --out /tmp/snr_sweep
# 2) 用现有 diag 脚本(Task 3 时的 /tmp/diag_probs.py 或等价物)对 5 个 wav 逐一测
#    当前 v3 部署 checkpoint 在【每轮语音结束点】的 P(TEXT_BEGIN), 记成表:
#    | wav | 轮1结束点P | 轮2结束点P | ... |
#    | snr_clean | | | |
#    | snr_20db  | | | |
#    | snr_15db  | | | |
#    | snr_10db  | | | |
#    | snr_05db  | | | |
```

**判读(拿到表后按此决策)**:
- clean 高、**SNR 15~20dB 就明显掉**(比如掉到 clean 的 1/3 以下)→ 噪声轴主导实锤,
  执行 Task 3 重训,预期收益大;
- clean 高、**SNR 5dB 仍稳** → 语音带噪不是瓶颈,**停,不要重训**;直接跳 Task 5 汇报,
  并检查:4A 锚定集是否真的进了训练数据?Task 6.0 checkpoint 扫描做了没有?
  (v3 方案里这两项对信道轴最重,别被增强绕开);
- clean 本身就低 → 实验前提不成立,换一条确认会开口的 wav 重来(先用 diag 扫几条选出来)。

## Task 3 重建数据 + 重训(仅当 Task 2 确认)

```bash
# ⚠️ 必须删旧产物(resume:auto 会捡旧 checkpoint 续训)
rm -rf zh_finetune/runtime/train_output/zh-full
rm -rf zh_finetune/runtime/data_full/work zh_finetune/runtime/data_full/train_jsonl

tmux new -s zhft
bash zh_finetune/run_all.sh --full --devices 1 \
     --data /path/合并后的data.jsonl --workers 64
# 数据不变(TTS 9k + aishell + ramc 同一个合并 jsonl), 变的只是构造:
# 噪声床默认已开(BED_PROB=0.5), 无需加参数; 想调占比用 --bed-prob(加在
# cons_online_data_zh.py 的调用处, run_all 未透传该参, 默认 0.5 即论文形态与
# v3 形态对半, 不建议改)。
```

起跑守门日志,**比 v3 多两行**,缺一即停下排查:

```
[延迟] 回复决策点滞后于语音结束: mean=180ms min=120ms max=240ms
[床] 噪声床样式 ~50% 条, 目标 SNR mean≈12.5dB range=[5,20]dB      ← 新
[响度] v3空档样式: 语音比空档高 ≥15dB ✓
[响度] bed床样式: 语音比空档高 4~21dB ✓                            ← 新(即实测SNR)
[flash] ✅   [ckpt] ✅
```

训练量与 v3 相同(~13-15k 对话、2 epoch、单卡 10-13 小时),中断重跑同命令续训。

## Task 4 验收(协议与 v3 Task 6 相同 + 一项加测)

1. **checkpoint 扫描**:对 train_output/zh-full 全部存档,用 held-out 实录测语音结束点
   P(TEXT_BEGIN),选真实表现最好的部署(不默认 final);
2. held-out 实录逐帧 P 表(重点指标);
3. 训练分布 wav 回归(每轮各开口一次);
4. **新增:抗噪加测** —— 把 Task 2 的 SNR 表在新 checkpoint 上重测一遍,
   对比重训前后:v4 成功的标志是 SNR 10~15dB 下 P 不再崩。

## Task 5 汇报清单(执行完发回)

- [ ] Task 2 的 SNR×P 表 + 判读结论(走了哪个分支)
- [ ] (若重训)Task 3 守门日志五行 + 首次 eval 的 val_moss_zh
- [ ] (若重训)Task 4 的 checkpoint 扫描表 + 重训前后 SNR 表对比 + 选定部署 ckpt
- [ ] 4A 锚定集状态确认:多少条实录进了训练?held-out 留了几条?
      (若从未做过 4A, 明确写"未做"——这是下一轮最优先补的)
