"""GQA flash-attn 运行时补丁(不改上游一行)。

背景: 上游 model.py:531-534 的 flash 快速路径要求 `n_query_groups == n_head`(纯 MHA),
而本模型是 GQA(16 头/2 组), 该分支永远不触发, 训练实际走
`repeat_interleave 扩展 k/v → F.scaled_dot_product_attention`(model.py:548-551, 619-621)。

本补丁把 `CausalSelfAttention.scaled_dot_product_attention` 换成 flash 版本:
  * 只接管训练热路径(mask is None ⇔ 因果全序列), 其余情形原样回退:
      - 推理 KV cache(mask 非 None)、capture_attn 可视化、softcapping、非 CUDA、
        非 fp16/bf16 —— 全部走原实现;
  * softmax_scale 显式镜像上游: 1/sqrt(attention_scores_scalar or head_size);
  * rope 后 q/k 可能被 fp32 cos/sin 提升精度 → cast 回 v.dtype
    (顺带一提: 上游自己的 flash 分支没做这个 cast, 真触发会直接报 mixed-dtype 错,
     侧面说明那段是没跑过的死代码);
  * flash_attn_func 输入要 (B,T,H,D)、上游方法约定返回 (B,T,H,D) —— transpose 进、
    直接返回, 无需 contiguous 拷贝。

用法:
    from zh_finetune.flash_patch import apply_patch
    apply_patch()   # True=已启用; False=flash-attn 不可用(自动回退, 不报错)

自检/基准(需 GPU):
    python zh_finetune/flash_patch.py          # 数值对齐校验 + fwd/bwd 微基准
"""
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_PATCH_FLAG = "_zh_flash_gqa_patched"


def flash_available() -> bool:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func  # noqa: F401
        return True
    except Exception:
        return False


def apply_patch() -> bool:
    """给 CausalSelfAttention.scaled_dot_product_attention 打 flash 补丁。幂等。"""
    if not flash_available():
        return False
    import torch
    from flash_attn.flash_attn_interface import flash_attn_func
    from src.audiointeraction.model import CausalSelfAttention

    if getattr(CausalSelfAttention, _PATCH_FLAG, False):
        return True
    _orig = CausalSelfAttention.scaled_dot_product_attention

    def _flash_sdpa(self, q, k, v, mask=None):
        cfg = self.config
        if (
            mask is None
            and not self.capture_attn
            and cfg.attention_logit_softcapping is None
            and q.device.type == "cuda"
            and v.dtype in (torch.float16, torch.bfloat16)
            and q.size(2) == k.size(2)          # Tq == Tk(训练全序列)
        ):
            scale = 1.0 / math.sqrt(cfg.attention_scores_scalar or cfg.head_size)
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
            if k.dtype != v.dtype:
                k = k.to(v.dtype)
            # (B, nh, T, hs) -> (B, T, nh, hs); 上游方法本就约定返回 (B, T, nh, hs)
            return flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, softmax_scale=scale, causal=True,
            )
        return _orig(self, q, k, v, mask)

    CausalSelfAttention.scaled_dot_product_attention = _flash_sdpa
    setattr(CausalSelfAttention, _PATCH_FLAG, True)
    return True


# ---------------- 自检: 数值对齐 + 微基准 ----------------

def _selfcheck():
    import types

    import torch
    from flash_attn.flash_attn_interface import flash_attn_func  # noqa: F401
    from src.audiointeraction.model import CausalSelfAttention

    assert apply_patch(), "flash-attn 不可用"
    patched = CausalSelfAttention.scaled_dot_product_attention
    # 原实现: 补丁内部闭包引用, 这里重新造一个"未打补丁"的调用途径
    import importlib
    import src.audiointeraction.model as m
    _orig_src = importlib.reload  # noqa: F841 (避免真的 reload, 用闭包里的 _orig 更直接)

    # 直接构造与真实模型同形状的张量: B=2, nh=16(已扩展), T=2048, hs=128
    torch.manual_seed(0)
    B, H, T, D = 2, 16, 2048, 128
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(B, H, T, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, H, T, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, H, T, D, device=dev, dtype=dt, requires_grad=True)
    self_ns = types.SimpleNamespace(
        config=types.SimpleNamespace(
            attention_scores_scalar=None, head_size=D, attention_logit_softcapping=None,
        ),
        capture_attn=False,
    )

    import torch.nn.functional as F

    def sdpa_ref(q, k, v):  # 镜像上游 else 分支
        scale = 1.0 / math.sqrt(D)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                           scale=scale, is_causal=True)
        return y.transpose(1, 2)

    with torch.no_grad():
        y_ref = sdpa_ref(q, k, v)
        y_fla = patched(self_ns, q, k, v, None)
    diff = (y_ref.float() - y_fla.float()).abs()
    print(f"[数值] max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}  (bf16 容差 ~2e-2)")
    assert diff.max().item() < 2e-2, "flash 与 SDPA 数值不一致!"

    def bench(fn, tag, iters=30):
        for _ in range(5):
            y = fn(q, k, v); loss = y.float().square().mean(); loss.backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True); t0.record()
        for _ in range(iters):
            y = fn(q, k, v); loss = y.float().square().mean(); loss.backward()
            q.grad = k.grad = v.grad = None
        t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / iters
        print(f"[基准] {tag}: {ms:.3f} ms/iter (fwd+bwd, B{B} H{H} T{T} D{D})")
        return ms

    ms_ref = bench(sdpa_ref, "SDPA(上游路径)")
    ms_fla = bench(lambda a, b, c: patched(self_ns, a, b, c, None), "flash-attn(补丁)")
    print(f"[基准] 单层注意力加速: {ms_ref / ms_fla:.2f}×  "
          f"(全模型收益按注意力占比折算, 序列越长收益越大)")


if __name__ == "__main__":
    _selfcheck()
