"""Actual native assignment and matched two-group training for CALFA group/codebook refinement."""
import torch

from kvquant import gpu_kmeans
from sglang.srt.mem_cache import vq_codebook


def encode(normalized, permutation, books):
    """Stored-scale normalization is already done across all eight heads."""
    assert normalized.shape[1:] == (128,)
    assert permutation.shape == (128,) and books.shape == (16, 256, 8)
    cb = books.half().contiguous()[None]
    # The real load_vq_codebook computes this reduction on CPU, then copies
    # it to CUDA. A CUDA sum differs by a few ulps and can change ties.
    squared = (.5 * cb.cpu().float().pow(2).sum(-1)).to(cb.device)
    reordered = normalized[:, permutation].contiguous()[:, None]
    # The native cache key contains data_ptr/shape, NOT a mutation version.
    # This offline process changes codebooks: invalidate before every encoding
    # so allocator pointer reuse cannot substitute an older candidate's C.
    vq_codebook._VQ_PAD_CACHE.clear()
    labels, scale = vq_codebook.vq_encode_single(
        reordered, cb, squared, pertoken_norm=False, pool_heads=False)
    assert torch.equal(scale, torch.ones_like(scale))
    return labels[:, 0].long()


def decode(normalized, stored_scale, channel_weight, permutation, books):
    labels = encode(normalized, permutation, books)
    groups = torch.arange(16, device=labels.device)[None]
    codewords = books.half()[groups, labels].float().reshape(-1, 128)
    reordered = codewords * stored_scale.float()[:, None] / channel_weight[permutation]
    decoded = torch.empty_like(reordered)
    decoded[:, permutation] = reordered
    assert torch.isfinite(decoded).all()
    return decoded, labels


def fit_two_groups(normalized, fisher, channel_weight, stored_scale,
                   permutation, groups, seed):
    """Same maximum 100 Lloyd iterations and native Fisher scalar as E0.

    Both swap and no-swap controls use this exact two-group batching/seed.
    Their batch maximum normalizer is explicit; this is not a reproduction
    of the parent's original 128-group random-number stream.
    """
    columns = torch.cat([permutation[g * 8:(g + 1) * 8] for g in groups])
    data = normalized[:, columns].reshape(-1, 2, 8).transpose(0, 1).contiguous()
    importance = fisher[:, columns].reshape(-1, 2, 8).transpose(0, 1).contiguous()
    w2 = channel_weight[columns].square().reshape(2, 8).clamp_min(1e-12)
    weight = (importance / w2[:, None]).sum(-1)
    weight *= stored_scale.float().square()[None]
    assert weight.max() > 0 and torch.isfinite(weight).all()
    weight /= weight.max()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    diagnostic = []
    gpu_kmeans._DIAGNOSTICS_OBSERVER = diagnostic.extend
    try:
        books, _ = gpu_kmeans.weighted_kmeans_batch(
            data, weight, 256, num_iters=100, fisher_init=False)
    finally:
        gpu_kmeans._DIAGNOSTICS_OBSERVER = None
    assert books.shape == (2, 256, 8) and books.dtype == torch.float32
    assert torch.isfinite(books).all() and torch.isfinite(books.half()).all()
    return books, diagnostic
