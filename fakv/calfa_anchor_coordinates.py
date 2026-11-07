"""CALFA-KV Anchor coordinate reference; not an attention or lifecycle implementation.

Canonical tensors end in [8 global KV heads, 128 channels]. Scale is the
actual affine-runtime FP16 value, treated as constant side information. Callers
must invert actual attention-space reconstruction (including real-position
RoPE and rounding) before passing reconstructed K into joint_error_score.
"""
import torch

COORDINATE_VERSION = "F_anchor_calfa_affine_z_v1"


def _validate(values, weights, permutation, scale16):
    if values.shape[-2:] != (8, 128):
        raise ValueError("Expected all 8 global KV heads and 128 channels")
    if weights.dtype != torch.float32 or weights.shape[-2:] != (8, 128):
        raise ValueError("W must be frozen FP32 [...,8,128]")
    if permutation.dtype != torch.int64 or permutation.shape != weights.shape:
        raise ValueError("P must match W with int64 indices")
    if scale16.dtype != torch.float16 or scale16.shape != values.shape[:-2]:
        raise ValueError("Scale must be the actual stored FP16 value per token")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise ValueError("Non-finite coordinate input")
    if not (weights > 0).all() or not (torch.isfinite(scale16) & (scale16 > 0)).all():
        raise ValueError("Nonpositive W or invalid stored scale")
    expected = torch.arange(128, device=permutation.device).expand_as(permutation)
    if not torch.equal(permutation.sort(-1).values, expected):
        raise ValueError("P is not a permutation")
    # Native CALFA-KV writes each original NeoX pair ADJACENT after P, not as
    # split halves: [frequency, frequency+64, next_frequency, ...].
    if not (permutation[..., 0::2] < 64).all() or not torch.equal(
        permutation[..., 1::2], permutation[..., 0::2] + 64
    ):
        raise ValueError("P must preserve native adjacent RoPE pairs and orientation")


def encode_at_fixed_scale(raw_k, weights, permutation, scale16):
    """z = ((x W) / s) P, with no recomputation or derivative of s."""
    _validate(raw_k, weights, permutation, scale16)
    scaled = raw_k.float() * weights / scale16.detach().float()[..., None, None]
    return scaled.gather(-1, permutation.expand(scaled.shape))


def inverse_coordinates(z_k, weights, permutation, scale16):
    """Algebraic coordinate inverse only; does not pretend to decode runtime KV."""
    _validate(z_k, weights, permutation, scale16)
    canonical = torch.zeros_like(z_k, dtype=torch.float32).scatter(
        -1, permutation.expand(z_k.shape), z_k.float()
    )
    return canonical * scale16.detach().float()[..., None, None] / weights


def signed_gradient_to_z(gradient_x, weights, permutation, scale16):
    """Apply inverse Jacobian to signed gradients BEFORE squaring/aggregation."""
    _validate(gradient_x, weights, permutation, scale16)
    transformed = gradient_x.float() / weights * scale16.detach().float()[..., None, None]
    return transformed.gather(-1, permutation.expand(transformed.shape))


def joint_error_score(raw_k, decoded_raw_k, raw_v, decoded_v,
                      weights, permutation, scale16, fisher_k, fisher_v):
    """Score actual decoded errors in z; never multiply by scale squared again.

    decoded_raw_k must already reflect every actual runtime rounding operation
    and inverse RoPE at the candidate's absolute position. Feeding an ideal
    normalized codeword difference would violate the caller contract.
    """
    if raw_k.shape != decoded_raw_k.shape or raw_v.shape != raw_k.shape or decoded_v.shape != raw_v.shape:
        raise ValueError("K/V and reconstructed tensors must have matching shapes")
    delta_k = encode_at_fixed_scale(raw_k.float() - decoded_raw_k.float(),
                                   weights, permutation, scale16)
    delta_v = raw_v.float() - decoded_v.float()
    for tensor in (delta_v, fisher_k, fisher_v):
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite score input")
    if (fisher_k < 0).any() or (fisher_v < 0).any():
        raise ValueError("Fisher weights must be nonnegative")
    return (fisher_k * delta_k.square() + fisher_v * delta_v.square()).sum(-1)


def regularize_joint_fisher(k, v, shrinkage=0.05, winsor_quantile=0.995):
    """Preserve the historical FAKV separate winsor + joint K/V estimator."""
    if k.shape != v.shape or k.shape[-2:] != (8, 128):
        raise ValueError("K/V Fisher shape mismatch")
    if not 0 <= shrinkage <= 1 or not 0 < winsor_quantile <= 1:
        raise ValueError("Invalid regularization parameters")
    if not torch.isfinite(k).all() or not torch.isfinite(v).all() or (k < 0).any() or (v < 0).any():
        raise ValueError("Invalid raw Fisher")
    if winsor_quantile < 1:
        k = torch.minimum(k, torch.quantile(k, winsor_quantile, dim=-1, keepdim=True))
        v = torch.minimum(v, torch.quantile(v, winsor_quantile, dim=-1, keepdim=True))
    joint_mean = 0.5 * (k.mean(-1, keepdim=True) + v.mean(-1, keepdim=True))
    if (joint_mean <= 0).any() or not torch.isfinite(joint_mean).all():
        raise ValueError("Empty or non-finite joint Fisher")
    k = (1 - shrinkage) * k + shrinkage * joint_mean
    v = (1 - shrinkage) * v + shrinkage * joint_mean
    normalizer = 0.5 * (k.mean(-1, keepdim=True) + v.mean(-1, keepdim=True))
    return (k / normalizer).float().contiguous(), (v / normalizer).float().contiguous()


def coordinate_identity(weights, permutation, center, radial):
    """Bind Fisher to final D/P/mu/A, including dtype and stored-scale contract."""
    import hashlib
    digest = hashlib.sha256(b"CALFA-KV:F_anchor_calfa_affine_z_v1:calfa_center_radial_v1")
    for name, tensor, dtype in (("D", weights, torch.float32), ("P", permutation, torch.int64),
                                ("mu", center, torch.float32), ("A", radial, torch.float32)):
        if tensor is None or tensor.shape != (36, 8, 128) or tensor.dtype != dtype:
            raise ValueError("Invalid final CALFA parameter " + name)
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite final CALFA parameter " + name)
        if name in ("D", "A") and not (value > 0).all():
            raise ValueError("Nonpositive final CALFA parameter " + name)
        if name == "P":
            wanted = torch.arange(128).expand_as(value)
            if not torch.equal(value.sort(-1).values, wanted):
                raise ValueError("P is not a complete per-head permutation")
            if not (value[..., ::2] < 64).all() or not torch.equal(value[..., 1::2], value[..., ::2]+64):
                raise ValueError("P splits or reverses native RoPE pairs")
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_quantized_center(natural_k, slots, hp_offset, center):
    """Restore mu only on quantized prefix rows; HP/Anchor rows are physical."""
    return torch.where((slots < hp_offset)[:, None, None],
                       natural_k.float()+center, natural_k.float()).to(natural_k.dtype)
