"""Direct projected attention-output objective; no KL or raw-K MSE selection."""
import math

import torch


def rotate_half(x):
    first, second = x.chunk(2, -1)
    return torch.cat((-second, first), -1)


def decode_fixed_indices(books, indices, scales, weights, permutations):
    """Continuous codeword subproblem with fixed actual hard assignments.

    No half cast/STE here. Deployment projection and reassignment are external.
    """
    heads, groups, _, group_dim = books.shape
    hi = torch.arange(heads, device=books.device)[None, :, None]
    gi = torch.arange(groups, device=books.device)[None, None, :]
    words = books[hi, gi, indices].reshape(-1, heads, groups * group_dim)
    wp = weights.gather(-1, permutations).to(books.dtype)
    transformed = words * scales.to(books.dtype)[:, None, None] / wp
    return torch.zeros_like(transformed).scatter(
        -1, permutations.expand_as(transformed), transformed)


def reference_state(q, raw_k, value, cos, sin, positions, projection,
                    recent=64, dtype=torch.float32):
    """BF16-source reference in explicitly shared arithmetic and coordinates."""
    assert q.shape[0] == raw_k.shape[0] == value.shape[0] == 1
    assert raw_k.shape == value.shape
    qr, kr = q.transpose(1, 2).to(dtype), raw_k.transpose(1, 2).to(dtype)
    c, s = cos[:, None].to(dtype), sin[:, None].to(dtype)
    qr = qr * c[:, :, positions] + rotate_half(qr) * s[:, :, positions]
    kr = kr * c + rotate_half(kr) * s
    ratio = qr.shape[1] // kr.shape[1]
    assert qr.shape[1] % kr.shape[1] == 0
    vv = value.transpose(1, 2).to(dtype).repeat_interleave(ratio, 1)
    logits = qr @ kr.repeat_interleave(ratio, 1).transpose(-1, -2) / math.sqrt(qr.shape[-1])
    tokens = torch.arange(kr.shape[2], device=q.device)[None, :]
    pos = torch.as_tensor(positions, device=q.device)[:, None]
    causal = (tokens <= pos)[None, None]
    history = (tokens < pos + 1 - recent)[None, None]
    probability = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    merged = (probability @ vv).transpose(1, 2).reshape(1, len(positions), -1)
    proj = projection.to(dtype)
    teacher = merged @ proj.T
    return dict(qr=qr, c=c, s=s, ratio=ratio, value=vv, logits=logits,
                causal=causal, history=history, teacher=teacher,
                projection=proj, positions=positions,
                energy=teacher.double().square().sum().detach())


def output_sse(decoded_k, reference):
    """Actual reconstructed K -> softmax -> raw V -> frozen output projection."""
    r = reference
    k = decoded_k.transpose(0, 1)[None].to(r["qr"].dtype)
    k = k * r["c"] + rotate_half(k) * r["s"]
    logits = r["qr"] @ k.repeat_interleave(r["ratio"], 1).transpose(-1, -2)
    logits = logits / math.sqrt(k.shape[-1])
    mixed = torch.where(r["history"], logits, r["logits"])
    probability = mixed.masked_fill(~r["causal"], -torch.inf).softmax(-1)
    merged = (probability @ r["value"]).transpose(1, 2).reshape(
        1, len(r["positions"]), -1)
    output = merged @ r["projection"].T
    residual = output - r["teacher"]
    return residual.double().square().sum(), residual


def self_checks(device="cpu"):
    generator = torch.Generator(device=device).manual_seed(210745926)
    def rand(shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float64)
    q, k, v = rand((1, 4, 4, 8)), rand((1, 9, 2, 8)), rand((1, 9, 2, 8))
    angle = rand((1, 9, 4))
    cos, sin = angle.cos().repeat(1, 1, 2), angle.sin().repeat(1, 1, 2)
    projection = rand((6, 32))
    ref = reference_state(q, k, v, cos, sin, [1, 3, 5, 8], projection,
                          recent=3, dtype=torch.float64)
    zero, _ = output_sse(k[0], ref)
    assert zero.item() == 0
    kd = k[0].clone()
    kd[8] += 100
    assert output_sse(kd, ref)[0].item() == 0
    books = rand((2, 2, 3, 4)).requires_grad_()
    indices = torch.randint(3, (9, 2, 2), generator=generator, device=device)
    permutation = torch.tensor([[3, 7, 0, 4, 2, 6, 1, 5]] * 2, device=device)
    weights = rand((2, 8)).abs() + .5
    scale = rand((9,)).abs() + .5
    def loss(c):
        decoded = decode_fixed_indices(c, indices, scale, weights, permutation)
        return output_sse(decoded, ref)[0] / ref["energy"]
    grad, = torch.autograd.grad(loss(books), books)
    direction = rand(books.shape)
    direction /= direction.norm()
    epsilon = 1e-5
    numerical = (loss(books.detach() + epsilon * direction) -
                 loss(books.detach() - epsilon * direction)) / (2 * epsilon)
    analytical = (grad * direction).sum()
    torch.testing.assert_close(numerical, analytical, rtol=2e-5, atol=1e-8)
    # Explicit scalar causal/GQA and original-coordinate output reconstruction.
    decoded = decode_fixed_indices(books.detach(), indices, scale, weights, permutation)
    heads = []
    pos = 8
    for h in range(4):
        qq = q[0, -1, h] * cos[0, pos] + rotate_half(q[0, -1, h]) * sin[0, pos]
        scores = []
        for t in range(pos + 1):
            kk = (decoded[t, h // 2] if t < pos + 1 - 3 else k[0, t, h // 2])
            kk = kk * cos[0, t] + rotate_half(kk) * sin[0, t]
            scores.append(qq.dot(kk) / math.sqrt(8))
        heads.append(torch.stack(scores).softmax(0) @ v[0, :, h // 2])
    projected = torch.cat(heads) @ projection.T
    _, residual = output_sse(decoded, ref)
    torch.testing.assert_close(projected - ref["teacher"][0, -1], residual[0, -1],
                               rtol=1e-10, atol=1e-10)
    return dict(passed=True, zero_error_exact=True, causal_recent_exact=True,
                fixed_assignment_directional_gradient=True,
                scalar_gqa_output_projection=True, inverse_w_scale_p_in_gradient=True,
                kl_or_raw_k_mse_objective=False)
