"""Read-only P/codebook diagnostics, in original K coordinates.

Native indices/codewords are supplied by the real KV pool. Attention below is
an explicit local reference, NOT the serving kernel or final vocabulary KL.
"""
import math

import torch


def rotate_half(x):
    first, second = x.chunk(2, -1)
    return torch.cat((-second, first), -1)


def masked_kl(reference, candidate, causal):
    """KL(reference || candidate), stable FP64 on valid causal logits only."""
    reference = reference.double().masked_fill(~causal, -torch.inf)
    candidate = candidate.double().masked_fill(~causal, -torch.inf)
    log_p = reference.log_softmax(-1)
    log_q = candidate.log_softmax(-1)
    probability = log_p.exp()
    # Avoid 0 * (inf - inf) even when masked positions have zero probability.
    difference = torch.where(causal, log_p - log_q, 0.0)
    kl = (probability * difference).sum(-1)
    assert torch.isfinite(kl).all() and (kl >= -1e-10).all()
    delta = torch.where(causal, candidate - reference, 0.0)
    centered = delta - (probability * delta).sum(-1, keepdim=True)
    quadratic = 0.5 * (probability * centered.square()).sum(-1)
    entropy = -(probability * torch.where(causal, log_p, 0.0)).sum(-1)
    return kl, quadratic, probability, log_q.exp(), entropy


def attention_diagnostics(q, raw_k, decoded_k, value, cos, sin, positions,
                          recent=64):
    """FP32 RoPE/GQA logits, FP64 softmax; fixed BF16 raw V, Recent unchanged.

    Both reference and reconstruction use the SAME FP32 rotary arithmetic,
    ensuring zero reconstruction error gives zero KL. No extra BF16 rounding
    of decoded K is introduced. This is a local mathematical diagnostic, not
    a claim of bitwise equivalence to fused serving attention.
    """
    assert raw_k.shape == decoded_k.shape == value.shape
    assert q.ndim == raw_k.ndim == 4 and q.shape[0] == raw_k.shape[0] == 1
    assert q.shape[1] == len(positions) and 0 <= recent <= raw_k.shape[1]
    assert q.shape[2] % raw_k.shape[2] == 0
    qh = q.transpose(1, 2).float()
    kr, kd = (x.transpose(1, 2).float() for x in (raw_k, decoded_k))
    c, s = cos[:, None].float(), sin[:, None].float()
    qr = qh * c[:, :, positions] + rotate_half(qh) * s[:, :, positions]
    kr, kd = (x * c + rotate_half(x) * s for x in (kr, kd))
    ratio = qr.shape[1] // kr.shape[1]
    kr, kd = (x.repeat_interleave(ratio, 1) for x in (kr, kd))
    original = qr @ kr.transpose(-1, -2) / math.sqrt(qr.shape[-1])
    compressed = qr @ kd.transpose(-1, -2) / math.sqrt(qr.shape[-1])
    token = torch.arange(kr.shape[2], device=q.device)[None, :]
    pos = torch.tensor(positions, device=q.device)[:, None]
    causal = (token <= pos)[None, None]
    history = (token < (pos + 1 - recent))[None, None]
    mixed = torch.where(history, compressed, original)
    kl, quadratic, p, phat, entropy = masked_kl(original, mixed, causal)
    values = value.transpose(1, 2).double().repeat_interleave(ratio, 1)
    output, altered = p @ values, phat @ values
    return dict(kl=kl[0].transpose(0, 1),
                quadratic_kl=quadratic[0].transpose(0, 1),
                reference_entropy=entropy[0].transpose(0, 1),
                output_sse=(altered - output).square().sum(-1)[0].T,
                output_signal=output.square().sum(-1)[0].T,
                query_positions=torch.tensor(positions, device=q.device))


class Moments:
    """Per-head centered covariance across token rows; FP64 sufficient stats."""
    def __init__(self, heads, channels, device):
        self.count = 0
        self.sum = torch.zeros(heads, channels, device=device, dtype=torch.float64)
        self.gram = torch.zeros(heads, channels, channels, device=device,
                                dtype=torch.float64)

    def add(self, value):
        x = value.transpose(0, 1).double()
        self.count += value.shape[0]
        self.sum += x.sum(1)
        self.gram += x.transpose(1, 2) @ x

    def finish(self):
        mean = self.sum / self.count
        covariance = self.gram / self.count - mean[:, :, None] * mean[:, None, :]
        variance = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        denominator = torch.sqrt(variance[:, :, None] * variance[:, None, :])
        correlation = torch.where(denominator > 1e-20,
                                  covariance / denominator.clamp_min(1e-20), 0)
        return dict(count=self.count, mean=mean.cpu(),
                    covariance=covariance.cpu(), correlation=correlation.cpu())


def codeword_summary(counts):
    assert counts.ndim == 3 and counts.shape[-1] == 256
    mass = counts.sum(-1, keepdim=True)
    assert (mass > 0).all()
    probability = counts.double() / mass
    entropy = -(probability * probability.clamp_min(1e-300).log2()).sum(-1)
    return dict(occupied=(counts > 0).sum(-1).tolist(),
                index_entropy_bits=entropy.tolist(),
                largest_cluster_fraction=probability.max(-1).values.tolist(),
                index_bits_stored=8,
                entropy_does_not_change_fixed_width_bpe=True)


def self_checks(device="cpu"):
    generator = torch.Generator(device=device).manual_seed(210745926)
    def rand(shape):
        return torch.randn(shape, generator=generator, device=device).bfloat16()
    q, k, v = rand((1, 2, 4, 8)), rand((1, 8, 2, 8)), rand((1, 8, 2, 8))
    angle = torch.randn((1, 8, 4), generator=generator, device=device) * .2
    cos, sin = angle.cos().repeat(1, 1, 2), angle.sin().repeat(1, 1, 2)
    exact = attention_diagnostics(q, k, k, v, cos, sin, [3, 7], 2)
    assert torch.equal(exact['kl'], torch.zeros_like(exact['kl']))
    assert torch.equal(exact['output_sse'], torch.zeros_like(exact['output_sse']))
    changed = k.clone()
    changed[:, -1] += 100
    future = attention_diagnostics(q, k, changed, v, cos, sin, [3, 7], 2)
    assert torch.equal(future['kl'], exact['kl'])
    changed = k.clone()
    changed[:, 0] += torch.arange(8, device=device).bfloat16()
    result = attention_diagnostics(q, k, changed, v, cos, sin, [3, 7], 2)
    assert (result['kl'] > 0).any()
    # Independent scalar query/head/reference with identical FP32 RoPE.
    for i, pos in enumerate((3, 7)):
        for h in range(4):
            kv = h // 2
            qq = q[0, i, h].float()
            qq = qq * cos[0, pos] + torch.cat((-qq[4:], qq[:4])) * sin[0, pos]
            logits, altered = [], []
            for t in range(pos + 1):
                pair = []
                for source in (k, changed if t < pos + 1 - 2 else k):
                    kk = source[0, t, kv].float()
                    kk = kk * cos[0, t] + torch.cat((-kk[4:], kk[:4])) * sin[0, t]
                    pair.append(qq.double().dot(kk.double()) / math.sqrt(8))
                logits.append(pair[0]); altered.append(pair[1])
            log_p = torch.stack(logits).log_softmax(0)
            log_q = torch.stack(altered).log_softmax(0)
            expected = (log_p.exp() * (log_p - log_q)).sum()
            torch.testing.assert_close(result['kl'][i, h], expected, atol=2e-6, rtol=2e-5)
    a = torch.randn((2, 7), generator=generator, device=device).double()
    causal = torch.arange(7, device=device)[None] < torch.tensor([4, 7], device=device)[:, None]
    kl, quad, *_ = masked_kl(a, a + 8.0, causal)
    assert float(kl.abs().max()) < 1e-12 and float(quad.abs().max()) < 1e-25
    x = rand((16, 2, 8)).float()
    moments = Moments(2, 8, device)
    moments.add(x[:5]); moments.add(x[5:])
    finished = moments.finish()
    centered = x.double() - x.double().mean(0)
    reference = torch.stack([centered[:, h].T @ centered[:, h] / 16 for h in range(2)])
    torch.testing.assert_close(finished['covariance'], reference.cpu(), atol=1e-12, rtol=1e-12)
    counts = torch.zeros(2, 3, 256, dtype=torch.int64)
    counts[..., 0] = 10
    assert codeword_summary(counts)['index_entropy_bits'] == [[0.0]*3]*2
    return dict(passed=True, device=device, zero_error_exact=True,
                causal_recent_ignored=True, independent_scalar_gqa_rope_kl=True,
                constant_logit_shift_invariance=True,
                partitioned_covariance_reference=True, collapsed_entropy_zero=True)
