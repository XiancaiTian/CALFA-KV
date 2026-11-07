"""Validate the actual loaded initialization pool against CALFA parent assets."""
from pathlib import Path
import hashlib
import torch


def verify_parent_pool(pool, assets):
    root=Path(assets)/'bundle'
    if (pool.layer_num,pool.head_num,pool.head_dim,pool.hp_prefix_tokens,pool.hp_recent_tokens) != (36,8,128,0,64):
        raise ValueError('CALFA training requires the full TP1/P0/Recent64 layout')
    if pool.hp_dtype != torch.bfloat16 or pool.scale_dtype != torch.float16:
        raise ValueError('CALFA training requires BF16 Recent and FP16 stored scale')
    blobs=[torch.load(root/n,map_location='cpu',weights_only=False) for n in ['codebook.pt','codebook_v.pt']]
    k,v=blobs
    if k.get('scale_contract') != 'calfa_fp16_scale_v1':
        raise ValueError('Initialize calibration from zero-center parent assets, not final affine books')
    if not pool._R_k_identity or not pool._R_v_identity or pool.shadow_enabled:
        raise ValueError('Parent pool must retain the fixed identity rotations and native history layout')
    for blob,codec in zip(blobs,[pool._vq,pool._vq_v]):
        expected=torch.stack([torch.stack([torch.stack(blob['codebooks'][l,h]) for h in range(8)]) for l in range(36)])
        if not torch.equal(codec.cb16.cpu(),expected) or codec.cb16.dtype != torch.float16:
            raise ValueError('Actual loaded K/V codebook differs from the declared parent')
        if codec.mean.any():
            raise ValueError('Parent encoding origin must be zero')
    if not torch.equal(pool._vq.channel_weights.cpu(),k['channel_weights']) or not torch.equal(pool._vq.channel_permutations.cpu(),k['channel_permutations']):
        raise ValueError('Actual loaded D/P differs from the parent')
    if pool._vq_v.pertoken_norm or not pool._vq.pool_heads_scale:
        raise ValueError('Parent requires shared K scale and unit V scale')
    return dict(passed=True,assets_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                   for n in ['codebook.pt','codebook_v.pt']})
