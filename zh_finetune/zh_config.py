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

# 块边界对齐:让每轮语音的"结束帧 % CHUNK_SIZE ∈ ALIGN_TARGET_MODS(每轮随机)"。
# mod=m → 决策点在语音结束后 (10-m)*40ms。取 {4,5,6,7} → 120~240ms(均值 ~180ms),
# 对应论文 half-chunk align δ=200ms;仍保证"结束后下一个 400ms 决策点内开口"。
# ⚠️ 教训(2026-07-09): 曾取恒定 8(=80ms), 与"空档噪声无增益控制"叠加导致
# 正样本零声学证据、零方差 → 训一个 epoch 后模型坍缩为永远沉默(P(TEXT_BEGIN)<1%)。
# 证据量(结束后的安静时长)和样本方差都是决策可学性的必要条件, 不要再钉死到 <=80ms。
ALIGN_TARGET_MODS = (4, 5, 6, 7)

# 空档噪声增益控制:把每段空档噪声衰减到「本对话语音 RMS − U(lo,hi) dB」。
# ⚠️ 同日教训: MS-SNSD 原生振幅拼接时噪声仅比语音低 ~3dB 且 21% 是人声类(Babble/Cafe),
# "静音"听起来像有人说话 → 模型学到"人声→保持沉默"、且语音结束无能量落差可依。
# 衰减后空档 = 远处环境声, 既有真实的"说完变安静"边界, 也保留抗噪鲁棒性。只衰减不放大。
NOISE_ATTEN_DB_RANGE = (18.0, 30.0)

# 空档噪声池默认剔除的"前景人声"类别(正则, 匹配文件名)。
# 衰减后的 Cafe/嘈杂环境声保留(远处人声嗡嗡是真实场景), 但 Babble(近距离多人清晰说话)
# 与 AirportAnnouncement(清晰广播词) 与用户语音过于相似, 默认从空档池剔除。
GAP_NOISE_EXCLUDE_PATTERN = r"Babble|AirportAnnouncement"

# 拼接处淡入淡出窗口(毫秒), 对应论文 fade window ω=20ms。
# 只对噪声段做边缘 fade(不动语音),消除硬拼接的咔哒声——避免模型把爆音当"说完了"的线索。
CROSSFADE_MS = 20

# 采样率/帧常量(与上游 load_audio.py 一致, 仅为可读性重复声明)
SAMPLE_RATE = 16000
FRAME_MS = 40                 # 1 encoder 帧 = 40ms = 640 samples
CHUNK_FRAMES = 10             # 1 chunk = 10 帧 = 400ms
