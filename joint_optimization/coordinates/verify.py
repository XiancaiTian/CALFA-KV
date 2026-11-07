#!/usr/bin/env python3
"""CPU checks for the CALFA coordinate optimization code and its one initialization asset set."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true', help='Also load trusted local W/P and books on CPU')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for item in manifest['files']:
        assert sha(root / item['path']) == item['sha256'], item['path']
    for item in manifest['reused_sources']:
        assert sha(root.parent / item['path']) == item['sha256'], item['path']
    import pair_groups as groups
    native = [c for pair in range(64) for c in (pair, pair + 64)]
    groups.validate([native] * 8)
    invalid = [c for g in range(0, 64, 4) for c in list(range(g,g+4))+list(range(g+64,g+68))]
    try:
        groups.validate([invalid] * 8)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonadjacent native RoPE pairs were accepted')
    variants = groups.partitions(native[:8], native[8:16])
    assert len({tuple(a) for a,b in variants}) == 35
    for a,b in variants:
        assert len(set(a+b)) == 16 and not set(a) & set(b)
        assert all(row[j+1] == row[j]+64 for row in (a,b) for j in range(0,8,2))
    for layer in range(36):
        for head in range(8):
            first = groups.blocks(layer,head,0);second = groups.blocks(layer,head,1)
            assert sorted(x for pair in first for x in pair) == list(range(16))
            assert sorted(x for pair in second for x in pair) == list(range(16))
            assert {frozenset(x) for x in first}.isdisjoint(frozenset(x) for x in second)
    split = groups.split()
    assert [len(split[k]) for k in split] == [36,12,48,16]
    assert not set(split['inner_fit']) & set(split['inner_selection'])
    assert not set(split['outer_fit']) & set(split['outer_validation'])
    parent = [dict(sse=10.,fisher_sse=20.)] * 2
    assert groups.decide(parent,[dict(sse=9.,fisher_sse=18.)]*2,.01)[0]
    assert not groups.decide(parent,[dict(sse=9.,fisher_sse=20.2)]*2,.01)[0]
    method = root.parents[1]
    sys.path[:0] = [str(method / 'backend/python'),str(root.parent / 'core')]
    import torch
    from calfa_scale import encode_features
    from grouping_rules import self_checks
    assert self_checks()['passed']
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(37,8,128,generator=generator)
    weights = torch.rand(8,128,generator=generator)+.1
    p = torch.tensor(native).expand(8,-1)
    normalized,scale = encode_features(raw,weights)
    permuted,scale_p = encode_features(raw,weights,p)
    chunked = torch.cat([encode_features(row,weights)[1] for row in raw.split(7)])
    assert torch.equal(scale.view(torch.int16),scale_p.view(torch.int16))
    assert torch.equal(scale.view(torch.int16),chunked.view(torch.int16))
    assert torch.equal(permuted,normalized.gather(-1,p.expand_as(normalized)))
    from attention_proxy import self_checks as check_attention
    assert check_attention()['passed']
    result = dict(passed=True,files_verified=len(manifest['files']),
                  reused_sources_verified=len(manifest['reused_sources']),
                  pair_constraints=True,two_pass_pairing=True,matched_pc_rules=True,
                  stored_FP16_scale_batch_and_permutation_invariant=True,attention_proxy=True)
    if args.assets:
        import numpy as np
        base = root / 'initialization'
        with (base/'wp.pkl').open('rb') as stream:
            wp = pickle.load(stream)
        assert wp['scale_contract'] == 'calfa_fp16_scale_v1'
        assert wp['frozen_input_manifest']['input_ids_sha256'] == json.loads((root/'config.json').read_text())['inputs']['input_ids_sha256']
        k = torch.load(base/'bundle/codebook.pt',map_location='cpu',weights_only=False)
        v = torch.load(base/'bundle/codebook_v.pt',map_location='cpu',weights_only=False)
        for layer in range(36):
            permutation = np.asarray(wp['perms'][layer])
            groups.validate(permutation.tolist())
            assert np.array_equal(k['channel_permutations'][layer].numpy(),permutation)
            assert np.array_equal(k['channel_weights'][layer].numpy(),np.asarray(wp['weights'][layer]))
        for book in (k,v):
            assert not book['mean'].any()
            assert len(book['codebooks']) == 36*8
            for tables in book['codebooks'].values():
                assert len(tables)==16
                assert all(t.shape==(256,8) and t.dtype==torch.float16 and torch.isfinite(t).all() for t in tables)
        result['single_initialization_asset_set'] = 'W/P, two G8/256 books, zero centers'
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
