#!/usr/bin/env python3
"""Validate final CALFA books and, optionally, their matching Anchor Fisher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/python'),str(ROOT/'fakv')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(bundle, anchor_manifest=None):
    import torch
    from calfa_anchor_coordinates import coordinate_identity
    k = torch.load(bundle/'codebook.pt', map_location='cpu', weights_only=False)
    v = torch.load(bundle/'codebook_v.pt', map_location='cpu', weights_only=False)
    assert k['scale_contract'] == 'calfa_center_radial_v1'
    identity = coordinate_identity(k['channel_weights'],k['channel_permutations'],k['affine_center'],k['affine_radial'])
    assert k['pertoken_norm'] and k['pool_heads_scale'] and k['perm_rope']
    assert not v['pertoken_norm'] and not v.get('pool_heads_scale',False)
    for side in (k,v):
        assert not side['mean'].any() and len(side['codebooks']) == 288
        for tables in side['codebooks'].values():
            assert len(tables) == 16
            assert all(t.shape == (256,8) and t.dtype == torch.float16 and torch.isfinite(t).all() for t in tables)
    for name in ('k_rotation_qqt_r_h_pbr.pt','v_rotation_sst_r_h_pbr.pt'):
        rotation = torch.load(bundle/name, map_location='cpu', weights_only=False)
        if isinstance(rotation,dict):
            rotation = torch.stack([rotation['layers'][i]['rotation'] for i in range(36)])
        assert isinstance(rotation,torch.Tensor),name
        assert torch.equal(rotation,torch.eye(128,dtype=rotation.dtype).expand_as(rotation)),name
    result = dict(passed=True,method='CALFA-KV',geometry='G8/256',coordinate_sha256=identity,
                  final_affine_books=True,anchor_fisher_checked=False)
    if anchor_manifest is not None:
        m = json.loads(anchor_manifest.read_text())
        assert (m['method'],m['G'],m['bits'],m['anchor_capacity']) == ('CALFA-KV',8,8,56)
        assert m['asset_ready'] and m['model_revision'] == 'cdbee75f17c01a7cc42f958dc650907174af0554'
        for name,item in m['assets'].items():
            path = (anchor_manifest.parent/item['path']).resolve()
            assert sha(path) == item['sha256'],name
            if name in ('k_bundle','v_bundle'):
                assert path == (bundle/('codebook.pt' if name=='k_bundle' else 'codebook_v.pt')).resolve()
        f = torch.load(anchor_manifest.parent/m['assets']['anchor_fisher']['path'],map_location='cpu',weights_only=False)
        assert f['coordinate_system'] == 'F_anchor_calfa_affine_z_v1'
        assert f['scale_contract'] == 'calfa_center_radial_v1' and f['coordinate_sha256'] == identity
        assert m['coordinate_sha256'] == identity and f['model_revision'] == m['model_revision']
        for n in ['k_weights','v_weights']:
            assert f[n].shape == (36,8,128) and f[n].dtype == torch.float32
            assert torch.isfinite(f[n]).all() and (f[n] > 0).all()
        result['anchor_fisher_checked'] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'assets/optimized')
    p.add_argument('--anchor-manifest',type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.bundle.resolve(), a.anchor_manifest.resolve() if a.anchor_manifest else None),indent=2))


if __name__ == '__main__':
    main()
