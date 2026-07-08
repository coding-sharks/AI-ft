#!/usr/bin/env bash
# ============================================================================
# zh_finetune 一键流程: 环境 → 权重 → 噪声 → 数据构造 → 训练
#
#   bash zh_finetune/run_all.sh --smoke                # 本机冒烟(替身音频, 1 卡, 12 步)
#   bash zh_finetune/run_all.sh --full                 # 正式训练(真实音频, 默认 8 卡)
#   可选: --data /path/data.jsonl   原始多轮数据(默认 <repo>/data.jsonl)
#         --devices N               覆盖 GPU 数
#         --audio-root PATH         真实音频根目录(默认 moss-zh 集群路径)
#         --skip-train              只准备数据不训练
#
# 幂等: 每个阶段先检查产物, 已存在则跳过; 中断后重跑即可续。
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$REPO/zh_finetune/runtime"
mkdir -p "$RT/logs"

MODE="" DATA="$REPO/data.jsonl" DEVICES="" SKIP_TRAIN=0 USE_FLASH=1 WORKERS=32
AUDIO_ROOT="/apdcephfs/private_giannishu/api_call/data_finetuning/moss-zh"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE=smoke; shift;;
    --full)  MODE=full;  shift;;
    --data)  DATA="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --audio-root) AUDIO_ROOT="$2"; shift 2;;
    --skip-train) SKIP_TRAIN=1; shift;;
    --no-flash) USE_FLASH=0; shift;;
    --workers) WORKERS="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done
[[ -z "$MODE" ]] && { echo "用法: bash zh_finetune/run_all.sh --smoke|--full [--data ...]"; exit 1; }
[[ -f "$DATA" ]] || { echo "找不到数据文件: $DATA"; exit 1; }
[[ "$MODE" == smoke ]] && DEVICES="${DEVICES:-1}" || DEVICES="${DEVICES:-8}"

banner() { echo; echo "=================== [$1] ==================="; }

# ---------------- 阶段 1: Python 环境 ----------------
banner "1/7 环境"
ENV_PREFIX="${AI_ENV_PREFIX:-$RT/env}"
PY="$ENV_PREFIX/bin/python"
env_ok() { "$PY" -c "import torch, transformers, lightning, whisper, librosa, soundfile, torchmetrics, safetensors" 2>/dev/null; }
if env_ok; then
  echo "环境已就绪: $ENV_PREFIX"
else
  CONDA=""
  for c in "$(command -v conda || true)" /opt/miniforge3/condabin/conda \
           "$HOME/miniconda3/condabin/conda" "$HOME/miniforge3/condabin/conda" \
           /opt/conda/condabin/conda; do
    [[ -n "$c" && -x "$c" ]] && { CONDA="$c"; break; }
  done
  if [[ -n "$CONDA" ]]; then
    echo "用 conda 建环境: $ENV_PREFIX"
    [[ -x "$PY" ]] || "$CONDA" create -y -p "$ENV_PREFIX" python=3.11
    "$CONDA" install -y -p "$ENV_PREFIX" -c conda-forge ffmpeg
  else
    echo "没找到 conda, 退回 python3 venv(需系统已有 ffmpeg!)"
    python3 -m venv "$ENV_PREFIX"
    command -v ffmpeg >/dev/null || echo "!! 警告: 系统无 ffmpeg, whisper.load_audio 会失败"
  fi
  "$ENV_PREFIX/bin/pip" install --upgrade pip
  "$ENV_PREFIX/bin/pip" install -r "$REPO/zh_finetune/requirements_zh.txt"
  env_ok || { echo "!! 环境自检失败"; exit 1; }
fi
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

# --- 可选加速: flash-attn(装失败不影响流程, 训练会自动回退 SDPA) ---
if [[ "$USE_FLASH" == 1 ]] && ! "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "尝试安装 flash-attn(可选; 编译需 10-30 分钟, 失败自动回退 SDPA)"
  TORCH_CU="$("$PY" -c 'import torch; print(torch.version.cuda or "")')"
  CH=""
  for d in "/usr/local/cuda-$TORCH_CU" /usr/local/cuda; do [[ -x "$d/bin/nvcc" ]] && { CH="$d"; break; }; done
  ARCH="$("$PY" -c 'import torch; c=torch.cuda.get_device_capability(0); print(f"{c[0]}.{c[1]}")' 2>/dev/null || echo 8.0)"
  if [[ -n "$CH" ]]; then
    "$ENV_PREFIX/bin/pip" install -q ninja packaging || true
    # 注意: $ENV_PREFIX/bin 必须在 PATH 里, 否则 torch 找不到 ninja 会退化成串行编译(慢一个数量级)
    ( export CUDA_HOME="$CH" PATH="$ENV_PREFIX/bin:$CH/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" \
             MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / 4 > 32 ? 32 : $(nproc) / 4 + 1 ))}"
      "$ENV_PREFIX/bin/pip" install flash-attn --no-build-isolation ) \
      && echo "flash-attn 安装成功" || echo "!! flash-attn 安装失败 → 训练将用 SDPA(功能等价)"
  else
    echo "!! 未找到与 torch(cu$TORCH_CU)匹配的 nvcc → 跳过 flash-attn, 用 SDPA"
  fi
fi

# ---------------- 阶段 2: 下载 checkpoint ----------------
banner "2/7 checkpoint"
CKPT="$REPO/checkpoints/audiointeraction"
ckpt_ok() {
  [[ -f "$CKPT/model_config.yaml" && -f "$CKPT/tokenizer.json" \
     && -f "$CKPT/audiointeraction_ChunkwisedEncoder.pth" \
     && -f "$CKPT/model.safetensors.index.json" \
     && -d "$CKPT/qwen25OmniConfig" ]] || return 1
  "$PY" - "$CKPT" <<'EOF'
import json, os, sys
c = sys.argv[1]
idx = json.load(open(os.path.join(c, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(c, s))]
sys.exit(1 if missing else 0)
EOF
}
if ckpt_ok; then
  echo "checkpoint 已就绪: $CKPT"
else
  echo "下载 zhifeixie/AudioInteraction (注意: download.py 里的 AudioInteraction-2M 是私有仓, 不用它)"
  "$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="zhifeixie/AudioInteraction", local_dir="$CKPT", max_workers=8)
EOF
  ckpt_ok || { echo "!! checkpoint 校验失败"; exit 1; }
fi

# ---------------- 阶段 3: safetensors → init.pt ----------------
banner "3/7 init.pt"
INIT="$RT/init.pt"
if [[ -f "$INIT" ]]; then
  echo "init.pt 已存在: $INIT"
else
  "$PY" "$REPO/zh_finetune/convert_safetensors_to_init.py" \
      --ckpt-dir "$CKPT" --out "$INIT" --verify
fi

# ---------------- 阶段 4: 噪声库 ----------------
banner "4/7 噪声库"
"$PY" "$REPO/zh_finetune/prepare_noise.py" --dest "$RT/noise" --min-files 20

# ---------------- 阶段 5: 数据格式转换 ----------------
banner "5/7 数据转换"
OUTD="$RT/data_$MODE"
ONLINE_IN="$OUTD/online_input.jsonl"
if [[ -f "$ONLINE_IN" ]]; then
  echo "已存在: $ONLINE_IN (删除该文件可强制重转)"
elif [[ "$MODE" == smoke ]]; then
  "$PY" "$REPO/zh_finetune/convert_to_online_input.py" \
      --in "$DATA" --out "$ONLINE_IN" \
      --replicate 64 \
      --stub-audio-glob "$REPO/assets/audio/*.wav" \
      --stub-audio-glob "$REPO/sample/*/*.wav" \
      --stub-dir "$OUTD/stub_audio"
else
  "$PY" "$REPO/zh_finetune/convert_to_online_input.py" \
      --in "$DATA" --out "$ONLINE_IN" \
      --audio-root "$AUDIO_ROOT" --check-audio
fi

# ---------------- 阶段 6: 四步数据构造(带延迟优化) ----------------
banner "6/7 数据构造"
TRAIN_JSONL="$OUTD/train_jsonl/train_online_zh.jsonl"
MAXSEQ=$([[ "$MODE" == smoke ]] && echo 2048 || echo 4096)
if [[ -f "$TRAIN_JSONL" ]]; then
  echo "已存在: $TRAIN_JSONL (删除 $OUTD/work 与该文件可强制重建)"
else
  cd "$REPO"
  "$PY" "$REPO/zh_finetune/cons_online_data_zh.py" \
      --input "$ONLINE_IN" --checkpoint-dir "$CKPT" \
      --work-dir "$OUTD/work" --out "$TRAIN_JSONL" \
      --noise-dir "$RT/noise" --max-seq "$MAXSEQ" --seed 1337 --workers "$WORKERS"
fi

# ---------------- 阶段 7: 训练 ----------------
banner "7/7 训练 ($MODE, ${DEVICES} 卡)"
[[ "$SKIP_TRAIN" == 1 ]] && { echo "--skip-train 已指定, 到此为止"; exit 0; }
export AI_CKPT="$CKPT" AI_RUNTIME="$RT" AI_DATA="$(dirname "$TRAIN_JSONL")" AI_DEVICES="$DEVICES"
export ZH_USE_FLASH="$USE_FLASH"
# 缓解显存碎片; 单卡时 train_zh.py 会自动附加激活重算补丁(见 ZH_GRAD_CKPT)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$REPO"
LOG="$RT/logs/train_${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "日志: $LOG"
"$PY" "$REPO/zh_finetune/train_zh.py" \
    --config "$REPO/zh_finetune/config_zh_${MODE}.yaml" 2>&1 | tee "$LOG"
echo
echo "✅ 全流程完成。checkpoint 在 $RT/train_output/zh-${MODE}/"
