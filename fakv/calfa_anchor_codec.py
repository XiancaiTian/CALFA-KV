"""Candidate-only reconstruction and scoring for TP1 CALFA-KV affine g8/bits8.

Reconstruction mirrors the native wide-G decoder: packed 8-bit codes, actual folded/unfolded FP16
codewords, the corresponding W inverse, block-16 hoisted trig, FP16 dot operand, THEN stored
FP16 scale. It is tested against optional exports from the actual decoder.
Only actual native migrants/prefill quantized rows may be passed; this file
does not alter the page-8 flush policy or store high-precision history.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct_candidates(
    Positions, Slots, Valid, KCodes, VCodes, KBook, VBook, Scale, Freq, WPerm,
    InvFreq, Center, OutK, OutV,
    KS0: tl.constexpr, KS1: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr,
    PREFIX: tl.constexpr,
    W_FOLDED: tl.constexpr,
    PAIR_MAJOR: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    active = tl.load(Valid + row).to(tl.int1)
    slot = tl.load(Slots + row)
    position = tl.load(Positions + row)
    pair = tl.arange(0, 64)
    group = pair // 4
    bit = group * 8
    word_index, shift = bit // 32, bit % 32
    word = tl.load(KCodes + slot * KS0 + head * KS1 + word_index,
                   mask=active, other=0).to(tl.int64) & 0xFFFFFFFF
    high = tl.load(KCodes + slot * KS0 + head * KS1 + word_index + 1,
                   mask=active & (shift + 8 > 32), other=0).to(tl.int64) & 0xFFFFFFFF
    code = ((word >> shift) | (high << (32 - shift))) & 255
    offset = (pair % 4) if PAIR_MAJOR else (pair % 4) * 2
    address = KBook + ((head * 16 + group) * 256 + code) * 8 + offset
    k1 = tl.load(address, mask=active, other=0).to(tl.float32)
    k2 = tl.load(address + (4 if PAIR_MAJOR else 1), mask=active, other=0).to(tl.float32)
    fidx = tl.load(Freq + head * 64 + pair)
    inv = tl.load(InvFreq + fidx).to(tl.float32)
    even_w = tl.load(WPerm + head * 128 + 2 * pair).to(tl.float32)
    odd_w = tl.load(WPerm + head * 128 + 2 * pair + 1).to(tl.float32)
    # MIN_BLOCK_KV=32 and BLOCK_N=16 make this base independent of KV splits.
    base = PREFIX + ((position - PREFIX) // 16) * 16
    ang_j = (position - base).to(tl.float32) * inv
    ang_b = base.to(tl.float32) * inv
    cos_j, sin_j = tl.cos(ang_j), tl.sin(ang_j)
    cos_b, sin_b = tl.cos(ang_b), tl.sin(ang_b)
    cs = cos_b * cos_j - sin_b * sin_j
    sn = sin_b * cos_j + cos_b * sin_j
    if W_FOLDED:
        k1u, k2u = k1, k2
    else:
        k1u, k2u = k1 / even_w, k2 / odd_w
    r1 = (k1u * cs - k2u * sn).to(tl.float16).to(tl.float32)
    r2 = (k2u * cs + k1u * sn).to(tl.float16).to(tl.float32)
    scale = tl.load(Scale + slot * SS0 + head * SS1, mask=active, other=1).to(tl.float32)
    mu1 = tl.load(Center + head * 128 + fidx).to(tl.float32)
    mu2 = tl.load(Center + head * 128 + 64 + fidx).to(tl.float32)
    m1 = tl.fma(mu1, cs, -(mu2 * sn)).to(tl.float16).to(tl.float32)
    m2 = tl.fma(mu2, cs, mu1 * sn).to(tl.float16).to(tl.float32)
    out_base = (row * 8 + head) * 128
    tl.store(OutK + out_base + fidx, tl.where(active, r1 * scale + m1, 0))
    tl.store(OutK + out_base + 64 + fidx, tl.where(active, r2 * scale + m2, 0))

    channel = tl.arange(0, 128)
    vg = channel // 8
    vbit = vg * 8
    vword_index, vshift = vbit // 32, vbit % 32
    vword = tl.load(VCodes + slot * VS0 + head * VS1 + vword_index,
                    mask=active, other=0).to(tl.int64) & 0xFFFFFFFF
    vhigh = tl.load(VCodes + slot * VS0 + head * VS1 + vword_index + 1,
                    mask=active & (vshift + 8 > 32), other=0).to(tl.int64) & 0xFFFFFFFF
    vcode = ((vword >> vshift) | (vhigh << (32 - vshift))) & 255
    value = tl.load(VBook + ((head * 16 + vg) * 256 + vcode) * 8 + channel % 8,
                    mask=active, other=0).to(tl.float32)
    tl.store(OutV + out_base + channel, value)


def reconstruct_candidates(positions, slots, valid, k_codes, v_codes, k_book,
                           v_book, scale16, freq_idx, w_perm, inv_freq, prefix=0,
                           w_folded=False, pair_major=False, center=None):
    """Return actual post-RoPE K and V for selected rows, not ideal centroids.

    inv_freq must be the native decoder's atan2(cos_sin_cache[1]) value.
    K/V arenas are the actual packed int32 caches; scale16 is the SAME stored
    FP16 tensor (head-broadcast strides are preserved, never recomputed).
    """
    if positions.ndim != 1 or slots.shape != positions.shape or valid.shape != positions.shape:
        raise ValueError("Candidate position/slot/valid shapes must match [N]")
    if positions.dtype != torch.int64 or slots.dtype != torch.int64 or valid.dtype != torch.bool:
        raise ValueError("Expected int64 positions/slots and bool validity")
    for codes in (k_codes, v_codes):
        if codes.dtype != torch.int32 or codes.shape[1:] != (8, 4) or not codes.is_contiguous():
            raise ValueError("Official g8/bits8 uses contiguous int32 [slots,8,4] arenas")
    for book in (k_book, v_book):
        if book.dtype != torch.float16 or book.shape != (8, 16, 256, 8) or not book.is_contiguous():
            raise ValueError("FP16 head_group g8/bits8 codebook required")
    if scale16.dtype != torch.float16 or scale16.shape != (k_codes.shape[0], 8, 1):
        raise ValueError("Actual FP16 shared scale view required")
    if freq_idx.dtype != torch.int64 or freq_idx.shape != (8, 64) or not freq_idx.is_contiguous():
        raise ValueError("Contiguous native freq_idx required")
    if w_perm.dtype != torch.float32 or w_perm.shape != (8, 128) or not w_perm.is_contiguous():
        raise ValueError("Contiguous native W[P] required")
    if inv_freq.dtype != torch.float32 or inv_freq.shape != (64,) or not inv_freq.is_contiguous():
        raise ValueError("Actual native atan2 inverse frequencies required")
    if center is None:
        raise ValueError("CALFA candidate reconstruction requires the frozen physical center")
    if center.shape != (8, 128) or center.dtype != torch.float32 or not center.is_contiguous():
        raise ValueError("Expected contiguous FP32 physical center [8,128]")
    if not all(x.is_cuda and x.device == positions.device for x in
               (positions, slots, valid, k_codes, v_codes, k_book, v_book,
                scale16, freq_idx, w_perm, inv_freq, center)):
        raise ValueError("Candidate reconstruction requires one CUDA device")
    torch._assert_async((~valid | ((positions >= prefix) & (slots >= 0)
                                  & (slots < k_codes.shape[0]) & (slots < v_codes.shape[0]))).all(),
                        "Candidate slot or absolute position outside native quantized region")
    out_k = torch.empty((positions.numel(), 8, 128), device=positions.device, dtype=torch.float32)
    out_v = torch.empty_like(out_k)
    if positions.numel():
        _reconstruct_candidates[(positions.numel(), 8)](
            positions, slots, valid, k_codes, v_codes, k_book, v_book, scale16,
            freq_idx, w_perm, inv_freq, center, out_k, out_v,
            k_codes.stride(0), k_codes.stride(1), v_codes.stride(0), v_codes.stride(1),
            scale16.stride(0), scale16.stride(1), prefix,
            w_folded, pair_major,
            num_warps=4,
        )
    return out_k, out_v


@triton.jit
def _score_actual_reconstruction(
    RawK, RawV, DecodedK, DecodedV, Positions, Valid, Scale16, W, P,
    FisherK, FisherV, CosSin, Scores, CS_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    active = tl.load(Valid + row).to(tl.int1)
    pos = tl.load(Positions + row)
    p = tl.load(P + head * 128 + d)
    frequency = p % 64
    cs = tl.load(CosSin + pos * CS_STRIDE + frequency, mask=active, other=1).to(tl.float32)
    sn = tl.load(CosSin + pos * CS_STRIDE + 64 + frequency, mask=active, other=0).to(tl.float32)
    base = (row * 8 + head) * 128
    r1 = tl.load(DecodedK + base + frequency, mask=active, other=0).to(tl.float32)
    r2 = tl.load(DecodedK + base + 64 + frequency, mask=active, other=0).to(tl.float32)
    # Invert the actual position's cached RoPE matrix, including its finite
    # precision norm. Native hoisted-trig discrepancies remain in the error.
    denominator = cs * cs + sn * sn
    # Default Triton "/" emits div.full.f32: its approximate quotient can
    # change the strict score ordering even with the identical reduction tree.
    # Match the independent FP32 reference's round-to-nearest division.
    xhat = tl.where(p < 64, tl.div_rn(r1 * cs + r2 * sn, denominator),
                   tl.div_rn(r2 * cs - r1 * sn, denominator))
    raw = tl.load(RawK + base + p, mask=active, other=0).to(tl.float32)
    weight = tl.load(W + head * 128 + p).to(tl.float32)
    scale = tl.load(Scale16 + row, mask=active, other=1).to(tl.float32)
    delta_k = tl.div_rn((raw - xhat) * weight, scale)
    v = tl.load(RawV + base + d, mask=active, other=0).to(tl.float32)
    vhat = tl.load(DecodedV + base + d, mask=active, other=0).to(tl.float32)
    delta_v = v - vhat
    fk = tl.load(FisherK + head * 128 + d).to(tl.float32)
    fv = tl.load(FisherV + head * 128 + d).to(tl.float32)
    score = tl.sum(fk * (delta_k * delta_k) + fv * (delta_v * delta_v), 0)
    tl.store(Scores + row * 8 + head, tl.where(active, score, -float("inf")))


def score_actual_reconstruction(raw_k, raw_v, decoded_k, decoded_v, positions,
                                valid, scale16, weights, permutation, fisher_k,
                                fisher_v, cos_sin_cache):
    """Canonical task-book score, including actual K decode and inverse RoPE."""
    n = positions.numel()
    for value in (raw_k, raw_v):
        if value.dtype != torch.bfloat16 or value.shape != (n, 8, 128) or not value.is_contiguous():
            raise ValueError("True BF16 canonical K/V [N,8,128] required")
    for value in (decoded_k, decoded_v):
        if value.dtype != torch.float32 or value.shape != raw_k.shape or not value.is_contiguous():
            raise ValueError("Actual contiguous FP32 decoded operands required")
    for value in (weights, fisher_k, fisher_v):
        if value.dtype != torch.float32 or value.shape != (8, 128) or not value.is_contiguous():
            raise ValueError("Contiguous FP32 coordinate/Fisher tensor required")
    if permutation.dtype != torch.int64 or permutation.shape != (8, 128) or not permutation.is_contiguous():
        raise ValueError("Contiguous int64 P required")
    if scale16.dtype != torch.float16 or scale16.shape != (n,) or not scale16.is_contiguous():
        raise ValueError("Actual shared FP16 scales [N] required")
    if positions.dtype != torch.int64 or positions.shape != (n,) or valid.dtype != torch.bool or valid.shape != (n,):
        raise ValueError("Int64 positions and boolean validity must be [N]")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 128 or cos_sin_cache.stride(1) != 1:
        raise ValueError("Actual native NeoX cos/sin cache required")
    tensors = (raw_k, raw_v, decoded_k, decoded_v, positions, valid, scale16,
               weights, permutation, fisher_k, fisher_v, cos_sin_cache)
    if not all(x.is_cuda and x.device == positions.device for x in tensors):
        raise ValueError("Scoring requires a single CUDA device")
    torch._assert_async((~valid | ((positions >= 0) & (positions < cos_sin_cache.shape[0]))).all(),
                        "Score position outside actual RoPE cache")
    torch._assert_async((~valid | (torch.isfinite(scale16) & (scale16 > 0))).all(),
                        "Invalid actual stored scale")
    scores = torch.empty((n, 8), device=positions.device, dtype=torch.float32)
    if n:
        _score_actual_reconstruction[(n, 8)](
            raw_k, raw_v, decoded_k, decoded_v, positions, valid, scale16,
            weights, permutation, fisher_k, fisher_v, cos_sin_cache, scores,
            cos_sin_cache.stride(0), num_warps=4, enable_fp_fusion=False,
        )
    torch._assert_async((~valid[:, None] | torch.isfinite(scores)).all(),
                        "Nonfinite actual Anchor score")
    return scores
