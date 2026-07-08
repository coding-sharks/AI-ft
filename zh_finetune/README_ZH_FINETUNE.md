# zh_finetune — Audio-Interaction 中文输出微调(一键流程)

> 目标:把 [Audio-Interaction](https://github.com/xzf-thu/Audio-Interaction)(清华 SoundFlow, 流式音频交互模型)
> 微调成**听中文语音 → 用中文文字回答**的助手,同时通过数据构造优化把回复延迟压到
> **语音结束后 ~80ms 的下一个决策点**。
>
> 本目录是**纯增量**:不改上游任何一行代码(中文 system prompt 等通过运行时补丁注入)。

## 快速开始

```bash
# 本机冒烟(无真实音频的机器; 用仓库自带 wav 当替身, 1 卡, 12 步, 验证全链路):
bash zh_finetune/run_all.sh --smoke

# 正式训练(音频所在集群; data.jsonl 为 9k+ 行原始多轮数据):
bash zh_finetune/run_all.sh --full --data /path/to/data.jsonl \
     [--devices 8] [--audio-root /apdcephfs/.../moss-zh]
```

就这一条命令。它幂等地依次完成:

| 阶段 | 做什么 | 产物(zh_finetune/runtime/) |
|---|---|---|
| 1 环境 | conda(缺则 venv)建 env + 装 requirements_zh.txt + ffmpeg | `env/` |
| 2 权重 | 从 HF **`zhifeixie/AudioInteraction`** 下载(⚠️ 上游 download.py 写的 `AudioInteraction-2M` 是私有仓,是错的) | `../checkpoints/audiointeraction/` |
| 3 转换 | 4 片 safetensors 合并 → litgpt `init.pt`(key 原生对齐,含 meta 校验) | `init.pt` |
| 4 噪声 | MS-SNSD noise_train(DNS-Challenge 同源, 128×16kHz wav);失败合成兜底 | `noise/` |
| 5 转格式 | 你的 `{turns:[{role,text,emotion}]}` → 上游要的 `{conversation:[{audio_path,assistant,emotion}]}`;smoke 模式用替身音频+复制 64 份 | `data_*/online_input.jsonl` |
| 6 构造 | 上游 4 步流水线 + **延迟优化**(见下)+ 中文 prompt 补丁 + 产物校验 | `data_*/train_jsonl/…` |
| 7 训练 | `train.py`(Lightning Fabric;多卡自动 FSDP) | `train_output/zh-*/` |

## 数据格式约定

输入 `data.jsonl` 每行:

```json
{"index": 18, "dialog": "moss-18", "turns": [
   {"turn_idx": 1, "role": "user",      "text": "怎么才能让自己更会自学啊?"},
   {"turn_idx": 2, "role": "assistant", "text": "想自学先定个小目标…", "emotion": "happy"},
   ...]}
```

- user/assistant 严格交替;user 轮的音频按 `{audio_root}/{dialog}/{dialog}_turn{turn_idx}.wav` 推断。
- `emotion` ∈ {happy, sad, angry, surprise, normal, urgent}(小写白名单;缺省/非法 → normal)。
- user 轮末尾没跟 assistant 回复 → 该轮被跳过并告警。

## 延迟优化(本目录的核心增值, 全在 cons_online_data_zh.py 的 step2 重写里)

上游把回复标签放在"包含语音结尾的那个 400ms 块"——名义延迟 0~400ms(平均 ~220ms),
且 **TTS wav 尾部自带的静音会被当成语音的一部分**,进一步推迟标签。我们做三件事:

1. **尾部静音修剪**(`librosa.effects.trim, top_db=40`):标签严格贴真实语音结尾;
2. **块边界对齐**(`ALIGN_TARGET_MOD=8`):微调每轮前导噪声长度,使语音结束帧 `%10==8`
   → 回复决策点固定在语音结束后 **(10-8)×40 = 80ms**(对应论文 half-chunk align δ=200ms 思路);
3. **20ms 边缘淡入淡出**(论文 fade window ω=20ms):只 fade 噪声段,消除拼接爆音。

step2 结束会打印 `[延迟] 回复决策点滞后于语音结束: mean=…ms max=…ms` 供核对。
用 `--no-align` 可关闭对齐做对照。

> 注意:这优化的是**模型学到的回复时机**。部署端实测延迟还取决于推理调度
> (论文的 FIFO 异步推理未开源;上游 infer 是同步的)与硬件解码速度。

## 训练配置

- `config_zh_smoke.yaml`:1 卡 / global 8 / micro 1 / seq 2048 / 12 步 —— 只为验证链路。
- `config_zh_full.yaml`:8 卡 FSDP / global 64 / micro 2 / seq 4096 / lr 2e-5(warmup 30 + cosine)/ 3 epochs。
  作者随 ckpt 发布的 `hyperparameters.yaml`(其最终 stage 存档)为 8 卡/64/2/epochs 2/min_lr 6e-5,
  我们因小数据(9k)把 lr 降到 2e-5 防训崩;观察 val loss 可上调至 5e-5。
- 全参微调(上游 train.py 语义),AdamW 只传 lr → **weight_decay=0.01(torch 默认)、β2=0.999**;
  论文用 β2=0.95,如需对齐请自行给 `instantiate_torch_optimizer` 传参。

## 训练后 → 推理

```bash
python zh_finetune/infer_online_zh.py \
    --checkpoint-dir checkpoints/audiointeraction \
    --lm zh_finetune/runtime/train_output/zh-full/final/lit_model.pth \
    --audio 你的测试.wav
```

它用与训练**同一个** `zh_config.ZH_SYSTEM_PROMPT` 建前缀(train/infer 前缀不一致会掉点),
LM 直接从 `.pt` 加载(上游 infer 只吃 safetensors 分片)。要打包给上游 `infer_online.py` 用,
可跑 `src/audiointeraction/finetune/extract_state_dict.py` 再自行重分片。

## 已核实的上游坑(踩过的都写这)

1. `download.py` 的 repo_id `AudioInteraction-2M` 是 401 私有仓 → 正确的是 `zhifeixie/AudioInteraction`;
2. README/`config.yaml` 注释里的入口 `finetune/full.py` 不存在 → 真入口 `src/audiointeraction/finetune/train.py`(须从仓库根 `-m` 方式跑);
3. HF 权重是 safetensors 分片,train.py 只吃单个 `.pt` → 本目录的 convert 脚本负责合并(作者自己的 `utils.load_model` 证明 key 是 litgpt 原生格式);
4. **上游 flash-attn 分支是死代码**:守卫要求 `n_query_groups == n_head`(纯 MHA),而本模型是 GQA(2 组/16 头)→ 永远不触发;且该分支没做 rope 后的 dtype 回转,真触发会报 mixed-dtype 错。**本目录用 `flash_patch.py` 运行时补丁让 GQA 训练热路径走 flash-attn**(见下节),flash-attn 装不上则自动回退 SDPA,不影响功能;
5. `litgpt` 包不需要装:全仓库 0 处 import(模型 vendored);
6. 超过 `max_seq_length` 的样本会让 `fill_in_audio_feature` 写入越界**直接崩训练**(不是安静截断)→ 构造末尾有超长过滤;
7. 样本数太少时 dataloader 的 `int(n×ratio)` 会把训练集切空 → smoke 模式先把 1 条复制 64 份;
8. 情感标签是白名单静默回退:不在 6 词表里的值(如 `neutral`/`excited`)会**无警告变成 normal** → 转换脚本做了显式校验;
9. **`cons_online_data.py` 发布版 import 就是坏的**:它 `from generate.base import resolve_checkpoint_paths`,但该函数实际在仓库根 `utils.py`(内部树没同步)→ 本目录在运行时把符号注入 `base` 模块后再 import 上游(见 `cons_online_data_zh.py::_lazy_imports`);
10. **`whisper.load_audio` 每段音频 fork 一个 ffmpeg**——conda 环境在网络盘(CephFS)时每次 exec 要跨网加载 libav* 动态库,实测 ~1.4s/次(5.3 万段 ≈ 25 小时),并行时进程集体卡 D 状态。→ 本目录运行时把 `whisper.load_audio` 换成 **soundfile 进程内直读**(`--no-sf-audio` 可回退),实测快 3 个数量级,且 **64/64 样本 token 序列与 ffmpeg 版逐位一致**(语音路径上游本就走 librosa;ffmpeg 仅用于 16k 噪声/拼接 wav,16k→16k 读取两种实现逐位相同)。step2 另有 `--workers N` 多进程并行(每记录独立播种,结果与并行度无关)。**环境必须常驻网络盘的场景(临时 GPU 容器)这是正解**:库只在进程启动加载一次,不再为每段音频付网络代价。

## flash-attn 加速(可选, 默认开)

- `run_all.sh` 阶段 1 会自动尝试安装 flash-attn(自动探测与 torch 匹配的 CUDA_HOME、
  按本机 GPU 架构单arch编译;失败**不阻塞流程**,训练自动回退 SDPA)。`--no-flash` 可整体关闭。
- ⚠️ **编译产物绑定 GPU 架构**:A100=sm80、H20/H100=sm90,一台机器编的 wheel 拷到
  不同架构的机器会报 `no kernel image available`,别跨架构搬 wheel。
- **机器上已有装好 flash-attn 的环境?** 用 `AI_ENV_PREFIX` 直接复用、零编译:
  `AI_ENV_PREFIX=/path/to/env bash zh_finetune/run_all.sh --full ...`
  (脚本会自检该 env 依赖是否齐全;transformers 需 4.57.x 以带 Qwen2.5-Omni 类)。
- 训练入口换成 `train_zh.py` = 上游 train.py + `flash_patch.py`:
  把 `CausalSelfAttention.scaled_dot_product_attention` 换成 flash 实现,
  **只接管训练热路径**(mask None 的因果全序列);KV-cache 推理、capture_attn、
  softcapping 等一律回退原实现。scale 显式镜像上游(1/√head_size);rope 提升的
  fp32 q/k cast 回 bf16(上游死代码没做这步)。
- 自检与微基准:`python zh_finetune/flash_patch.py`(数值对齐断言 + fwd/bwd 计时)。
- **实测(1×A100, flash-attn 2.8.3.post1, sm80)**:
  - 数值:max|Δ|=9.8e-4(bf16 容差 1/20 以内);smoke 复训 loss 轨迹与 SDPA **逐位一致**
    (final val 0.452/0.453 vs 0.452/0.453),数值等价成立;
  - 速度:单层注意力 fwd+bwd **1.07×**(T2048);端到端 iter 323.5ms vs 325ms(**~0.5%**,
    seq2048 时注意力只占整步 ~20%,收益被 MLP/优化器稀释)。seq4096(full 配置)注意力
    占比翻倍,预期端到端 ~2-5%;H20 算力弱、FA 相对收益略高。
  - 结论:**免费但小**的加速——机器已有 flash-attn(如 H20 用 AI_ENV_PREFIX 复用)就开着;
    需要现场编译且赶时间时,`--no-flash` 跳过毫不可惜。
- ⚠️ 编译提速要点(已写进 run_all.sh):`$ENV_PREFIX/bin` 必须在 PATH 里,否则 torch 找不到
  ninja 会**静默退化成串行编译**(128 核机器上 1 小时 vs 并行 7 分钟)。

## 单卡训练(H20 / 只有一张卡时必读)

上游 `train.py` **只在多卡(FSDP)时启用激活重算**,单卡没有任何省显存机制 →
3B 全参 + seq4096 单卡需要 **100GB+ 显存,必 OOM**(96G 的 H20 也不够)。

解法已内置:`train_zh.py` 在 `--devices 1` 时**自动**给 `Block.forward` 打
`torch.utils.checkpoint` 激活重算补丁(日志见 `[ckpt] ✅ 已启用`;
`ZH_GRAD_CKPT=0/1` 可强制关/开;多卡时自动让位给 FSDP 自带重算)。

**实测(1×A100-80G, full 几何 seq4096/micro2/重算开):Peak Memory = 72.95 GB** —
96G H20 余量 ~23G,`config_zh_full.yaml` 无需改动。代价:步时约 +30%。

单卡 full 命令:
```bash
bash zh_finetune/run_all.sh --full --devices 1 --data /path/9k.jsonl
```
单卡 H20 上 9k×3epoch 估 **~10-15 小时**,建议 tmux/nohup 挂后台。
另: run_all 已默认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。

## 冒烟测试结果(2026-07-07, 1×A100-80G, 已跑通 ✅)

`bash zh_finetune/run_all.sh --smoke` 端到端一次通过(阶段 1-7 全绿):

- **数据**:1 行 data.jsonl → 64 副本 / 320 配对(替身音频 8 个);构造后 64/64 样本保留、
  0 超长、8/8 特征对齐抽查通过。
- **延迟优化生效**:`回复决策点滞后于语音结束: mean=80ms max=80ms (align=on)`
  —— 每轮恒定 80ms(上游原版 40~400ms 抖动 + TTS 尾静音拖后)。
- **训练**(12 optimizer steps, seq 2048, micro 1, lr 2e-5 warmup4+cosine):
  loss 5.79 → **0.45**(train/val 同步下降, 64 副本记忆型冒烟符合预期);
  iter ~325ms;**峰值显存 75.9GB**(单卡无 FSDP 无激活重算,故 smoke 用 seq2048/micro1;
  full 模式 8 卡 FSDP+激活重算,单卡占用会低得多)。
- **产物**:`runtime/train_output/zh-smoke/final/lit_model.pth`(13GB, 纯模型)
  + `step-000010/`(含优化器状态, 可续训)+ TensorBoard 日志。
- 冒烟产物可整目录删除回收空间:`rm -rf zh_finetune/runtime/{train_output,data_smoke}`。
