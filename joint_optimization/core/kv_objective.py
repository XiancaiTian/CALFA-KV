"""Two-sided attention-output objective with the frozen query-wise Recent mask."""
import math
import torch
from attention_objective import output_sse, reference_state, rotate_half


def residual_components(decoded_k, decoded_v, reference):
    """Return K-only residual and V's increment under reconstructed-K attention.

    The teacher remains BF16-source K/V in shared reference arithmetic. Both
    reconstructions affect history only; Recent/future locations stay exact.
    This is a mathematical reference, not a claim of native kernel agreement.
    """
    r = reference
    k = decoded_k.transpose(0, 1)[None].to(r['qr'].dtype)
    k = k * r['c'] + rotate_half(k) * r['s']
    logits = r['qr'] @ k.repeat_interleave(r['ratio'], 1).transpose(-1, -2)
    logits = logits / math.sqrt(k.shape[-1])
    mixed = torch.where(r['history'], logits, r['logits'])
    probability = mixed.masked_fill(~r['causal'], -torch.inf).softmax(-1)
    decoded_value = decoded_v.transpose(0, 1)[None].to(r['value'].dtype)
    decoded_value = decoded_value.repeat_interleave(r['ratio'], 1)
    assert decoded_value.shape == r['value'].shape
    base = (probability @ r['value']).transpose(1, 2).reshape(1, len(r['positions']), -1)
    k_residual = base @ r['projection'].T - r['teacher']
    increment = ((probability * r['history']) @ (decoded_value - r['value']))
    increment = increment.transpose(1, 2).reshape(1, len(r['positions']), -1)
    v_increment = increment @ r['projection'].T
    return k_residual, v_increment


def kv_output_sse(decoded_k, decoded_v, reference):
    k_residual, v_increment = residual_components(decoded_k, decoded_v, reference)
    residual = k_residual + v_increment
    return residual.double().square().sum(), residual


def decomposition(decoded_k, decoded_v, reference):
    k_residual, v_increment = residual_components(decoded_k, decoded_v, reference)
    k, v = k_residual.double(), v_increment.double()
    k_sse = k.square().sum()
    v_sse = v.square().sum()
    cross = 2 * (k * v).sum()
    joint_sse = (k_residual + v_increment).double().square().sum()
    reconstructed = k_sse + v_sse + cross
    tolerance = 1e-5 if k_residual.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(joint_sse, reconstructed, rtol=tolerance, atol=tolerance)
    return dict(k_only_sse=float(k_sse), v_at_reconstructed_k_sse=float(v_sse),
                cross_term=float(cross), joint_sse=float(joint_sse),
                energy=float(reference['energy']))


def self_checks():
    gen = torch.Generator().manual_seed(8117)
    def rand(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64)
    q, k, v = rand((1, 4, 4, 8)), rand((1, 9, 2, 8)), rand((1, 9, 2, 8))
    angle = rand((1, 9, 4))
    cos, sin = angle.cos().repeat(1, 1, 2), angle.sin().repeat(1, 1, 2)
    positions = [1, 3, 5, 8]
    projection = rand((6, 32))
    ref = reference_state(q, k, v, cos, sin, positions, projection, recent=3, dtype=torch.float64)
    assert kv_output_sse(k[0], v[0], ref)[0] == 0
    kd, vd = (k[0] + .15 * rand(k[0].shape)), (v[0] + .13 * rand(v[0].shape))
    old_loss, old_residual = output_sse(kd, ref)
    same_loss, same_residual = kv_output_sse(kd, v[0], ref)
    assert torch.equal(old_loss, same_loss) and torch.equal(old_residual, same_residual)
    _, residual = kv_output_sse(kd, vd, ref)
    # Independently build every causal score and GQA value contribution.
    for qi, pos in enumerate(positions):
        heads = []
        for h in range(4):
            qq = q[0, qi, h] * cos[0, pos] + rotate_half(q[0, qi, h]) * sin[0, pos]
            scores, values = [], []
            for t in range(pos + 1):
                history = t < pos + 1 - 3
                kk = kd[t, h // 2] if history else k[0, t, h // 2]
                vv = vd[t, h // 2] if history else v[0, t, h // 2]
                kr = kk * cos[0, t] + rotate_half(kk) * sin[0, t]
                scores.append(qq.dot(kr) / math.sqrt(8))
                values.append(vv)
            heads.append(torch.stack(scores).softmax(0) @ torch.stack(values))
        actual = torch.cat(heads) @ projection.T - ref['teacher'][0, qi]
        torch.testing.assert_close(actual, residual[0, qi], rtol=1e-11, atol=1e-11)
    for side in ('k', 'v'):
        variable = (kd if side == 'k' else vd).clone().requires_grad_()
        direction = rand(variable.shape); direction /= direction.norm()
        def objective(x):
            return kv_output_sse(x if side == 'k' else kd, x if side == 'v' else vd, ref)[0] / ref['energy']
        gradient, = torch.autograd.grad(objective(variable), variable)
        step = 1e-5
        fd = (objective(variable.detach() + step * direction) - objective(variable.detach() - step * direction)) / (2 * step)
        torch.testing.assert_close(fd, (gradient * direction).sum(), rtol=2e-5, atol=1e-9)
    for side in ('k', 'v'):
        a, b = kd.clone(), vd.clone()
        (a if side == 'k' else b)[8] += 100
        assert torch.equal(kv_output_sse(a, b, ref)[1], residual)
    parts = decomposition(kd, vd, ref)
    # Concrete counterexample: better K-only reconstruction can be worse jointly.
    q2 = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
    k2 = torch.zeros((1, 2, 1, 2), dtype=torch.float64)
    v2 = torch.tensor([[[[1., 0.]], [[-1., 0.]]]], dtype=torch.float64)
    ref2 = reference_state(q2, k2, v2, torch.ones((1, 2, 2), dtype=torch.float64),
                           torch.zeros((1, 2, 2), dtype=torch.float64), [1],
                           torch.tensor([[1., 0.]], dtype=torch.float64), recent=0, dtype=torch.float64)
    va = v2[0].clone(); va[:, :, 0] -= .6
    records = []
    for prob in (.6, .7):
        ka = k2[0].clone(); ka[0, 0, 0] = math.log(prob / (1 - prob)) * math.sqrt(2)
        records.append(dict(probability=prob, k_only_sse=float(output_sse(ka, ref2)[0]),
                            joint_sse=float(kv_output_sse(ka, va, ref2)[0])))
    assert records[0]['k_only_sse'] < records[1]['k_only_sse']
    assert records[0]['joint_sse'] > records[1]['joint_sse']
    return dict(passed=True,scalar_causal_gqa_reference=True,k_and_v_directional_gradients=True,
                zero_error_exact=True,bf16_v_recovers_k_only_bitwise=True,
                future_and_recent_unchanged=True,cross_term_identity=True,
                synthetic_ranking_reversal=records,actual_calibration_claim=False,
                native_kernel_claim=False,accuracy_claim=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_checks()))
