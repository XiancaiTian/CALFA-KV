"""Finite-rate grouping kernel shared by both coordinate-search passes in CALFA-KV."""
import hashlib
import json
from pathlib import Path
import time

import torch
from kvquant import gpu_kmeans
from kvquant.frozen_training import load_layer
from calfa_scale import encode_features
import pair_groups as structure
from attention_proxy import attention_outputs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data, exclusive=False):
    if exclusive:
        with Path(path).open('x') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
    else:
        tmp = Path(path).with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)


def indices(windows, end, stride):
    return torch.tensor([w * 2048 + t for w in windows for t in range(0, end, stride)], device='cuda')


def gather(x, rows, definitions):
    h = torch.tensor([h for h, _ in definitions], device=x.device)
    c = torch.tensor([c for _, c in definitions], device=x.device)
    return x[rows[None, :, None], h[:, None, None], c[:, None, :]].contiguous()


def group_weights(d, rows, definitions):
    w = torch.stack([d['w'][h, c] for h, c in definitions])
    f = gather(d['fisher'], rows, definitions)
    return (f / w[:, None].square().clamp_min(1e-12)).sum(-1) * d['scale'][rows].float().square()[None]


def proxy_fit(x, weights, seed, iterations=50):
    """Common token-row initialization for all partitions; native FP16 assignment.

    Deliberately a bounded proxy, not the final plain-kmeans++ deployment trainer.
    Equal seeds select identical token IDs across groups and execution batches.
    """
    generator = torch.Generator(device=x.device).manual_seed(seed)
    rows = torch.randperm(x.shape[1], generator=generator, device=x.device)[:256]
    initial = x[:, rows].clone()
    books = initial.clone()
    assignment = gpu_kmeans.get_dist_argmin_half_batched(8)
    for _ in range(iterations):
        labels = assignment(x.half(), books.half()).long()
        count = torch.zeros(x.shape[0], 256, device=x.device)
        total = torch.zeros_like(books)
        count.scatter_add_(1, labels, weights)
        total.scatter_add_(1, labels[..., None].expand(-1, -1, 8), weights[..., None] * x)
        new = total / (count[..., None] + 1e-8)
        books = torch.where((new == 0).all(-1)[..., None], initial, new)
    assert torch.isfinite(books).all()
    return books.half()


def evaluate(d, rows, definitions, books):
    z = gather(d['z'], rows, definitions)
    labels = gpu_kmeans.get_dist_argmin_half_batched(8)(z.half(), books.half()).long()
    code = books.gather(1, labels[..., None].expand(-1, -1, 8)).float()
    w = torch.stack([d['w'][h, c] for h, c in definitions])
    restored = code * d['scale'][rows].float()[None, :, None] / w[:, None]
    error = (restored - gather(d['raw'], rows, definitions)).double().square()
    fisher = gather(d['fisher'], rows, definitions).double()
    return [dict(sse=float(e.sum()), fisher_sse=float((e*f).sum()),
                 fisher_mass=float(f.sum()), elements=e.numel()) for e, f in zip(error, fisher)]


def fit_score(d, definitions, fit, selection, seed, normalizer, proxy, log, context):
    scores, books_all = [], []
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    for start in range(0, len(definitions), 32):
        batch = definitions[start:start+32]
        x = gather(d['z'], fit, batch)
        weights = group_weights(d, fit, batch) / normalizer
        assert torch.isfinite(weights).all() and (weights.sum(-1) > 0).all()
        if proxy:
            books = proxy_fit(x, weights, seed)
        else:
            books, _ = gpu_kmeans.weighted_kmeans_batch(x, weights, 256, num_iters=100, fisher_init=False)
            books = books.half()
        rows = evaluate(d, selection, batch, books)
        scores.extend(rows); books_all.append(books.cpu())
        with log.open('a') as handle:
            handle.write(json.dumps(dict(**context, seed=seed, start=start, codebooks=len(batch),
                fit_rows=fit.numel(), selection_rows=selection.numel(), proxy=proxy,
                max_iterations=50 if proxy else 100, definitions=batch, scores=rows)) + '\n')
    return scores, torch.cat(books_all)


def total(rows):
    return {name: sum(r[name] for r in rows) for name in ('sse', 'fisher_sse', 'fisher_mass', 'elements')}


def definitions(permutation):
    return [(h, permutation[h][8*g:8*g+8]) for h in range(8) for g in range(16)]


def attention_score(config, layer, d, permutation, books):
    defs = definitions(permutation)
    windows, scores = config['split']['outer_validation'], []
    books = books.cuda()
    for window in windows:
        base = Path(config['q_data']) / ('window_%03d' % window)
        meta = read(base / 'complete.json'); record = meta['layers'][layer]
        assert record['layer'] == layer and record['kv_exact']
        assert record['original_kv_sha256'] == d['identities'][window]['file_sha256']
        assert sha(base / record['file']) == record['sha256']
        assert sha(base / 'rotary.pt') == meta['rotary_sha256']
        data = torch.load(base / record['file'], map_location='cuda', weights_only=True)
        rotary = torch.load(base / 'rotary.pt', map_location='cuda', weights_only=True)
        rows = torch.arange(window*2048, (window+1)*2048, device='cuda')
        labels = gpu_kmeans.get_dist_argmin_half_batched(8)(gather(d['z'], rows, defs).half(), books).long()
        decoded = books.gather(1, labels[..., None].expand(-1, -1, 8)).float()
        raw_restored = torch.empty_like(d['raw'][rows])
        for index, (h, cols) in enumerate(defs):
            raw_restored[:, h, cols] = decoded[index] * d['scale'][rows].float()[:, None] / d['w'][h, cols]
        pos = data['query_positions']
        assert pos == list(range(127, 2048, 128))
        ref, altered = attention_outputs(data['q_pre_rope'], d['raw'][rows][None],
            raw_restored[None], d['value'][rows][None], rotary['cos'], rotary['sin'], pos)
        diff = (altered - ref).double().square()
        scores.append(dict(window=window, sse=float(diff.sum()), elements=diff.numel(),
                           signal=float(ref.double().square().sum())))
    return dict(sse=sum(r['sse'] for r in scores), elements=sum(r['elements'] for r in scores),
                signal=sum(r['signal'] for r in scores), by_window=scores,
                scope='BF16 teacher-forced local decode proxy, fixed raw V, pre-output projection; not native kernel')


def search_layer(config, layer, baseline):
    root = Path(config['output']) / ('layer_%02d' % layer)
    root.mkdir(); started = time.monotonic()
    tensors, identities = load_layer(config['features'], layer, config['input_ids_sha256'])
    raw = tensors['k_feature'].reshape(-1, 8, 128).float().cuda()
    fisher = tensors['k_grad_sq_fp32'].reshape(-1, 8, 128).cuda()
    value = tensors['v_feature'].reshape(-1, 8, 128).cuda(); del tensors
    w = torch.as_tensor(baseline['weights'][layer], device='cuda', dtype=torch.float32)
    z, scale = encode_features(raw, w)
    d = dict(raw=raw, fisher=fisher, value=value, z=z, scale=scale, w=w, identities=identities)
    original = baseline['perms'][layer].tolist(); current = [list(row) for row in original]
    structure.validate(original)
    fit = indices(config['split']['inner_fit'], 2048, 4)
    select = indices(config['split']['inner_selection'], 1984, 8)
    old_defs = definitions(original)
    normalizer = max(float(group_weights(d, fit, old_defs[i:i+32]).max()) for i in range(0, 128, 32))
    log = root / 'fit_diagnostics.jsonl'
    state = dict(layer=layer, complete=False, input_ids_sha256=config['input_ids_sha256'],
                 feature_identities=identities, original_permutation=original, blocks=[])
    # All eight disjoint blocks are searched; the outer 16 windows remain untouched here.
    for block_index in range(8):
        defs, mappings = [], []
        for head in range(8):
            g1, g2 = structure.blocks(layer, head, config['pass_index'])[block_index]
            first, second = original[head][8*g1:8*g1+8], original[head][8*g2:8*g2+8]
            variants = [(first, second)] + structure.partitions(first, second)
            begin = len(defs)
            for a, b in variants:
                defs.extend([(head, a), (head, b)])
            mappings.append(dict(head=head, groups=[g1, g2], begin=begin, variants=variants))
        trials = [fit_score(d, defs, fit, select, seed, normalizer, True, log,
                           dict(stage='inner_search', block=block_index))[0] for seed in structure.SEEDS]
        choices = []
        for mapping in mappings:
            begin = mapping['begin']; before = [total(r[begin:begin+2]) for r in trials]
            best, best_score = 0, 1.
            for index in range(1, 36):
                after = [total(r[begin+2*index:begin+2*index+2]) for r in trials]
                accepted, score = structure.decide(before, after, config['local_margin'])
                if accepted and score < best_score:
                    best, best_score = index, score
            h = mapping['head']; g1, g2 = mapping['groups']
            a, b = mapping['variants'][best]
            # Unordered partition: assign orientation with greater old-channel overlap.
            if len(set(a) & set(original[h][8*g1:8*g1+8])) < len(set(b) & set(original[h][8*g1:8*g1+8])):
                a, b = b, a
            if best:
                current[h][8*g1:8*g1+8] = a; current[h][8*g2:8*g2+8] = b
            choices.append(dict(head=h, groups=[g1, g2], selected_index=best,
                                balanced_ratio=best_score, proposed_channels=[a, b]))
        state['blocks'].append(dict(block=block_index, choices=choices))
        save(root / 'progress.json', state)
        print(json.dumps(dict(layer=layer, block=block_index, accepted=sum(c['selected_index']>0 for c in choices))), flush=True)
    structure.validate(current)
    # One outer decision after complete search; full-row plain native training for both controls.
    fit = indices(config['split']['outer_fit'], 2048, 1)
    select = indices(config['split']['outer_validation'], 1984, 1)
    normalizer = max(float(group_weights(d, fit, old_defs[i:i+32]).max()) for i in range(0, 128, 32))
    comparisons, saved = [], {}
    for seed in structure.SEEDS:
        row = dict(seed=seed)
        for label, permutation in (('before', original), ('proposed', current)):
            metrics, books = fit_score(d, definitions(permutation), fit, select, seed, normalizer,
                                      False, log, dict(stage='outer_final', variant=label))
            row[label] = dict(reconstruction=total(metrics),
                              attention=attention_score(config, layer, d, permutation, books))
            saved[(seed, label)] = books
        comparisons.append(row)
    accepted, balanced = structure.decide([r['before']['reconstruction'] for r in comparisons],
        [r['proposed']['reconstruction'] for r in comparisons], config['outer_margin'])
    attention_ok = all(r['proposed']['attention']['sse'] <= 1.005*r['before']['attention']['sse'] for r in comparisons)
    state.update(complete=True, outer_comparisons=comparisons, outer_balanced_ratio=balanced,
                 reconstruction_accepted=accepted, attention_accepted=attention_ok,
                 layer_accepted=accepted and attention_ok, proposed_permutation=current,
                 final_permutation=current if accepted and attention_ok else original,
                 duration_seconds=time.monotonic()-started, config_sha256=config['_config_sha256'],
                 diagnostics_sha256=sha(log), calibration_only=True, no_deployment_or_accuracy_claim=True)
    torch.save(saved, root / 'outer_books.pt')
    state['outer_books_sha256'] = sha(root / 'outer_books.pt')
    save(root / 'complete.json', state, exclusive=True)
    print(json.dumps({k:state[k] for k in ('layer','layer_accepted','outer_balanced_ratio','duration_seconds')}), flush=True)
