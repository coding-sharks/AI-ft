"""zh_finetune 共享常量 — 中文微调的单一事实来源。

训练数据构造(cons_online_data_zh.py)和推理(infer_online_zh.py)都从这里读
系统提示词,保证 train/infer 前缀完全一致(这是硬要求:不一致会掉点)。
"""

# 中文系统提示词(替换上游英文 DEFAULT_SYSTEM_PROMPT / SYSTEM_PROMPT)。
# 语义上对齐原提示词的职责描述,但要求中文输出。
ZH_SYSTEM_PROMPT = (
    "你是一个乐于助人的语音助手。请认真听用户的语音,并用简体中文简洁、口语化地回答。"
    "如果音频里没有问题或不需要回应,请保持安静。"
)

# ---- 延迟优化(数据构造期)----
# 语音尾部静音修剪阈值(librosa.effects.trim top_db)。
# TTS 合成音频尾部往往带 100~500ms 静音;不修剪的话,模型会把"wav 结尾"当"说完",
# 学出额外等待 → 响应变慢。40dB 对干净 TTS 是保守安全值。
TRIM_TOP_DB = 40
# 修剪后至少保留的时长(秒),防止极端修剪把整条剪没。
TRIM_MIN_KEEP_S = 0.3

# 块边界对齐:让每轮语音的"结束帧 % CHUNK_SIZE == ALIGN_TARGET_MOD"。
# CHUNK_SIZE=10(400ms), 取 8 → 决策点固定在语音结束后 (10-8)*40ms = 80ms,
# 而不是平均 ~220ms、最差 400ms。对应论文 Table 11 的 half-chunk align δ=200ms 思路。
ALIGN_TARGET_MOD = 8

# 拼接处淡入淡出窗口(毫秒), 对应论文 fade window ω=20ms。
# 只对噪声段做边缘 fade(不动语音),消除硬拼接的咔哒声——避免模型把爆音当"说完了"的线索。
CROSSFADE_MS = 20

# 采样率/帧常量(与上游 load_audio.py 一致, 仅为可读性重复声明)
SAMPLE_RATE = 16000
FRAME_MS = 40                 # 1 encoder 帧 = 40ms = 640 samples
CHUNK_FRAMES = 10             # 1 chunk = 10 帧 = 400ms
