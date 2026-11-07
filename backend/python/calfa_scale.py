"""The calfa_fp16_scale_v1 K scale contract; no model or serving imports.

Canonical layout is [..., 8 global KV heads, 128 channels]. Reduction is an
explicit adjacent-pair FP32 tree over head-major channels (10 additions levels).
It does not depend on token batch size, head_order, permutation or a backend's
choice of reduction kernel. Every caller consumes the returned FP16 value.
"""
import torch

SCALE_CONTRACT = "calfa_fp16_scale_v1"
REDUCTION_ORDER = "head_major_1024_adjacent_pair_tree_fp32_v1"
SCALE_FLOOR = 1e-4


def _require(condition, message):
    # CUDA device assertions are capture-compatible; unlike .item(), these do not
    # synchronize or copy a live CUDA value to the CPU during graph capture.
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def stored_scale(weighted_k):
    """Freeze one FP16 RMS per token, before P, from FP32 W-weighted K."""
    if weighted_k.dtype != torch.float32 or weighted_k.shape[-2:] != (8, 128):
        raise ValueError("calfa_fp16_scale_v1 requires FP32 [...,8,128] weighted K")
    _require(torch.isfinite(weighted_k).all(), "nonfinite weighted K")
    sums = weighted_k.square().reshape(*weighted_k.shape[:-2], 1024)
    for _ in range(10):
        sums = sums[..., 0::2] + sums[..., 1::2]
    rms = (sums[..., 0] / 1024.0).sqrt()
    _require(torch.isfinite(rms).all(), "nonfinite FP32 K RMS")
    scale16 = rms.clamp_min(SCALE_FLOOR).to(torch.float16)
    _require((torch.isfinite(scale16) & (scale16 > 0)).all(),
             "K RMS cannot be stored as finite positive FP16")
    return scale16


def normalize_weighted_k(weighted_k):
    scale16 = stored_scale(weighted_k)
    return weighted_k / scale16.float()[..., None, None], scale16


def encode_features(raw_k, weights, permutation=None):
    """W -> freeze FP16 RMS -> divide by that value -> optional P."""
    if weights.shape != (8, 128) or weights.dtype != torch.float32:
        raise ValueError("W must be the frozen FP32 [8,128] asset")
    weighted = raw_k.float() * weights
    z, scale16 = normalize_weighted_k(weighted)
    if permutation is not None:
        if permutation.shape != (8, 128) or permutation.dtype != torch.int64:
            raise ValueError("P must be int64 [8,128]")
        z = z.gather(-1, permutation.expand(z.shape))
    return z, scale16


def reconstruct_scaled(codewords, scale16):
    if codewords.dtype != torch.float16 or scale16.dtype != torch.float16:
        raise ValueError("runtime codewords and stored scale must both be FP16")
    return codewords.float() * scale16.float()[..., None, None]


def encode_with_stored_scale(raw_k, weights, permutation, scale16):
    """Deferred HP demotion consumes frozen side information; NEVER recompute RMS.

    weights/permutation may broadcast over leading layer/token axes. Raw K is
    retained only for the SAME HP slots, so mapped BF16 rounding is not inverted.
    """
    if raw_k.shape[-2:] != (8, 128) or weights.shape[-2:] != (8, 128):
        raise ValueError("calfa_fp16_scale_v1 requires all 8 global heads")
    if weights.dtype != torch.float32 or permutation.dtype != torch.int64:
        raise ValueError("frozen W/P dtype mismatch")
    if scale16.dtype != torch.float16 or scale16.shape != raw_k.shape[:-2]:
        raise ValueError("stored FP16 scale shape mismatch")
    _require((torch.isfinite(scale16) & (scale16 > 0)).all(), "invalid stored K scale")
    weighted = raw_k.float() * weights
    _require(torch.isfinite(weighted).all(), "nonfinite weighted K on cache demotion")
    normalized = weighted / scale16.float()[..., None, None]
    return normalized.gather(-1, permutation.expand(normalized.shape))
