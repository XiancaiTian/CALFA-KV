"""Local teacher-forced decode proxy. This is not the native serving attention kernel."""
import math
import torch


def rotate(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def attention_outputs(q, raw_k, decoded_k, value, cos, sin, positions, recent=64):
    # Inputs in original head/channel coordinates; emulate BF16 RoPE arithmetic.
    q = q.transpose(1, 2).to(torch.bfloat16)
    raw_k = raw_k.transpose(1, 2).to(torch.bfloat16)
    decoded_k = decoded_k.transpose(1, 2).to(torch.bfloat16)
    cos, sin = cos[:, None], sin[:, None]
    q = (q * cos[:, :, positions] + rotate(q) * sin[:, :, positions]).float()
    kr = (raw_k * cos + rotate(raw_k) * sin).float()
    kd = (decoded_k * cos + rotate(decoded_k) * sin).float()
    ratio = q.shape[1] // kr.shape[1]
    kr, kd = (x.repeat_interleave(ratio, 1) for x in (kr, kd))
    v = value.transpose(1, 2).float().repeat_interleave(ratio, 1)
    scale = math.sqrt(q.shape[-1])
    raw_logits = q @ kr.transpose(-1, -2) / scale
    quant_logits = q @ kd.transpose(-1, -2) / scale
    token = torch.arange(kr.shape[2], device=q.device)[None, :]
    pos = torch.tensor(positions, device=q.device)[:, None]
    # The selected queries are all page-8 flush boundaries: exactly Recent64 HP slots.
    history = token < (pos + 1 - recent)
    causal = token <= pos
    mixed = torch.where(history[None, None], quant_logits, raw_logits)
    raw_prob = raw_logits.masked_fill(~causal[None, None], -torch.inf).softmax(-1)
    mixed_prob = mixed.masked_fill(~causal[None, None], -torch.inf).softmax(-1)
    return raw_prob @ v, mixed_prob @ v


def self_checks():
    generator = torch.Generator().manual_seed(210745926)
    q = torch.randn(1, 2, 4, 8, generator=generator).bfloat16()
    k = torch.randn(1, 8, 2, 8, generator=generator).bfloat16()
    v = torch.randn(1, 8, 2, 8, generator=generator).bfloat16()
    cos, sin = torch.ones(1, 8, 8).bfloat16(), torch.zeros(1, 8, 8).bfloat16()
    ref, exact = attention_outputs(q, k, k, v, cos, sin, [3, 7], 2)
    assert torch.equal(ref, exact)
    changed = k.clone(); changed[:, -1] += 100
    _, result = attention_outputs(q, k, changed, v, cos, sin, [3, 7], 2)
    assert torch.equal(ref, result), 'future/current HP keys must not be quantized'
    changed = k.clone(); changed[:, 0] += torch.arange(8).bfloat16()
    _, result = attention_outputs(q, k, changed, v, cos, sin, [3, 7], 2)
    assert not torch.equal(ref, result)
    for h in range(4):
        for i, end in enumerate((3, 7)):
            logits = k[0, :end+1, h//2].double() @ q[0, i, h].double() / math.sqrt(8)
            expected = logits.softmax(0) @ v[0, :end+1, h//2].double()
            torch.testing.assert_close(ref[0, h, i], expected.float(), atol=2e-6, rtol=2e-5)
    return dict(passed=True, zero_error_exact=True, high_precision_and_causality=True,
                independent_fp64_GQA_reference=True)
