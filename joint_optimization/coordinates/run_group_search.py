#!/usr/bin/env python3
"""Run either internal finite-rate pass of the CALFA coordinate optimization."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--pass-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--baseline-wp', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(36)))
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    if recipe['experiment'] != 'calfa_coordinates':
        parser.error('Expected CALFA coordinate configuration')
    if args.output.exists():
        parser.error('--output must be a new directory')
    if len(set(args.layers)) != len(args.layers) or any(x not in range(36) for x in args.layers):
        parser.error('--layers must contain distinct layer numbers from 0 to 35')
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / 'backend/python'), str(root / 'backend/kvquant')]
    import torch
    import finite_rate_groups as core
    import pair_groups
    policy = recipe['finite_rate_grouping']
    expected = dict(seeds=list(pair_groups.SEEDS), proxy_iterations=50, final_iterations=100,
                    inner_fit_stride=4, inner_selection_stride=8, local_margin=.01, outer_margin=.0025)
    for key, value in expected.items():
        if policy[key] != value:
            parser.error('Changed numerical policy: ' + key)
    if policy['split'] != pair_groups.split():
        parser.error('Changed calibration split')
    baseline = pickle.loads(args.baseline_wp.read_bytes())
    if baseline['scale_contract'] != 'calfa_fp16_scale_v1':
        parser.error('The grouping input must use calfa_fp16_scale_v1')
    identity = recipe['inputs']['input_ids_sha256']
    if baseline['frozen_input_manifest']['input_ids_sha256'] != identity:
        parser.error('Baseline and calibration inputs differ')
    for layer in range(36):
        pair_groups.validate(baseline['perms'][layer].tolist())
    queries = core.read(args.queries / 'complete.json')
    if not queries.get('passed') or queries['input_ids_sha256'] != identity:
        parser.error('Queries do not match the admitted raw calibration inputs')
    if queries.get('numeric_identity', {}).get('rotary_inv_freq_sha256') != recipe['inputs']['rotary_inv_freq_sha256']:
        parser.error('Query capture RoPE frequency identity differs from the pinned model')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('Expose exactly one CUDA GPU for a grouping worker')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = dict(policy, pass_index=args.pass_index, baseline_wp=str(args.baseline_wp.resolve()),
                  baseline_wp_sha256=core.sha(args.baseline_wp), features=str(args.features.resolve()),
                  q_data=str(args.queries.resolve()), output=str(args.output.resolve()),
                  input_ids_sha256=identity, layers=args.layers)
    config['_config_sha256'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True)
    core.save(args.output / 'run_config.json', config, exclusive=True)
    with torch.no_grad():
        for layer in args.layers:
            core.search_layer(config, layer, baseline)
            torch.cuda.empty_cache()
    core.save(args.output / 'complete.json', dict(complete=True, layers=args.layers,
              pass_index=args.pass_index, config_sha256=config['_config_sha256']), exclusive=True)


if __name__ == '__main__':
    main()
