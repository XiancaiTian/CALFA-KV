"""Centered single-FP16-scale codec, separately versioned from calfa_fp16_scale_v1."""
import torch
from calfa_scale import stored_scale, _require

SCALE_CONTRACT = 'calfa_center_radial_v1'


def validate_parameters(center, radial, weights, permutation):
    if center.shape != weights.shape or radial.shape != weights.shape or permutation.shape != weights.shape:
        raise ValueError('Affine center/A/D/P shape mismatch')
    if center.dtype != torch.float32 or radial.dtype != torch.float32 or weights.dtype != torch.float32 or permutation.dtype != torch.int64:
        raise ValueError('Affine center/A/D/P dtype mismatch')
    if weights.shape[-2:] != (8,128):
        raise ValueError('Affine codec requires TP1 global eight KV heads')


def freeze_scale(raw_k, center, radial):
    if raw_k.shape[-2:] != (8,128) or center.shape[-2:] != (8,128) or radial.shape[-2:] != (8,128):
        raise ValueError('Affine scale requires all global heads')
    if center.dtype != torch.float32 or radial.dtype != torch.float32:
        raise ValueError('Affine scale requires frozen FP32 center/A')
    return stored_scale((raw_k.float()-center)*radial)


def encode_with_stored_scale(raw_k, weights, permutation, scale16, center):
    if raw_k.shape[-2:] != (8,128) or weights.shape[-2:] != (8,128) or center.shape[-2:] != (8,128):
        raise ValueError('Affine encode requires all global heads')
    if weights.dtype != torch.float32 or center.dtype != torch.float32 or permutation.dtype != torch.int64:
        raise ValueError('Affine frozen parameters dtype mismatch')
    if scale16.dtype != torch.float16 or scale16.shape != raw_k.shape[:-2]:
        raise ValueError('Affine scale must be the once-stored FP16 value')
    _require((torch.isfinite(scale16) & (scale16>0)).all(),'Invalid stored affine FP16 scale')
    z=(raw_k.float()-center)*weights/scale16.float()[...,None,None]
    _require(torch.isfinite(z).all(),'Nonfinite affine encoding features')
    return z.gather(-1,permutation.expand(z.shape))


def encode_features(raw_k, center, radial, weights, permutation):
    validate_parameters(center,radial,weights,permutation)
    n=freeze_scale(raw_k,center,radial)
    return encode_with_stored_scale(raw_k,weights,permutation,n,center),n
