#!/usr/bin/env python3
"""Calibrate Anchor-only Fisher after freezing CALFA D/P/mu/A and K/V books."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/python'),str(ROOT/'fakv')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local pinned Qwen3-4B-Instruct-2507 checkpoint')
    parser.add_argument('--bundle', type=Path, default=ROOT/'assets/optimized')
    parser.add_argument('--input-ids', type=Path, default=ROOT/'joint_optimization/coordinates/inputs/input_ids_chat2056.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-windows', type=int, default=0, help='0 = all 64 frozen windows; record subsets explicitly')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from calfa_affine import freeze_scale
    from calfa_anchor_coordinates import coordinate_identity, signed_gradient_to_z, regularize_joint_fisher
    blob = torch.load(args.bundle/'codebook.pt', map_location='cpu', weights_only=False)
    if blob.get('scale_contract') != 'calfa_center_radial_v1':
        raise ValueError('Use final CALFA joint assets, not zero-center initialization books')
    D, P, mu, A = [blob[k].contiguous() for k in ('channel_weights','channel_permutations','affine_center','affine_radial')]
    identity = coordinate_identity(D, P, mu, A)
    records = json.loads(args.input_ids.read_text())
    if args.max_windows < 0:
        parser.error('--max-windows must be nonnegative')
    if args.max_windows:
        records = records[:args.max_windows]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                attn_implementation='eager', local_files_only=True).to(args.device).eval()
    c = model.config
    theta = getattr(c, 'rope_theta', None) or getattr(c, 'rope_parameters', {}).get('rope_theta')
    if (c.num_hidden_layers,c.num_key_value_heads,c.num_attention_heads,getattr(c,'head_dim',128)) != (36,8,32,128) or theta != 5000000:
        raise ValueError('Wrong model geometry or RoPE theta; expected Qwen3-4B-Instruct-2507 / 5M')
    model.requires_grad_(False)
    # Create the graph at embeddings without learning any model parameter.
    handles = [model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))]
    captured = {}
    def capture(layer, side):
        def hook(module, inputs, output):
            if not output.requires_grad:
                raise RuntimeError('Calibration activation is disconnected from the loss')
            output.retain_grad()
            captured[layer, side] = output
        return hook
    for layer, block in enumerate(model.model.layers):
        handles.append(block.self_attn.k_norm.register_forward_hook(capture(layer, 'K')))
        handles.append(block.self_attn.v_proj.register_forward_hook(capture(layer, 'V')))
    total_k = torch.zeros((36,8,128), dtype=torch.float64)
    total_v = torch.zeros_like(total_k)
    tokens = 0
    try:
        for index, row in enumerate(records):
            ids = row['input_ids'] if isinstance(row, dict) else row
            ids = torch.tensor(ids, dtype=torch.long, device=args.device).reshape(1,-1)
            if ids.shape[1] < 2:
                raise ValueError('Calibration window needs at least two tokens')
            captured.clear()
            logits = model(ids, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1,logits.shape[-1]), ids[:,1:].reshape(-1))
            loss.backward()
            for layer in range(36):
                key, value = captured[layer,'K'], captured[layer,'V']
                raw = key.detach().reshape(-1,8,128)
                scale = freeze_scale(raw, mu[layer].to(raw.device), A[layer].to(raw.device))
                # Transform SIGNED post-k_norm/pre-RoPE gradients before squaring.
                gk = signed_gradient_to_z(key.grad.reshape(-1,8,128).float(),
                          D[layer].to(raw.device), P[layer].to(raw.device), scale)
                gv = value.grad.reshape(-1,8,128).float()
                total_k[layer] += gk.double().square().sum(0).cpu()
                total_v[layer] += gv.double().square().sum(0).cpu()
            tokens += ids.numel()
            print(json.dumps(dict(window=index, tokens=ids.numel(), loss=float(loss.detach()))), flush=True)
            del logits,loss,key,value,raw,gk,gv
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()
    if tokens == 0:
        raise ValueError('Empty calibration inputs')
    fk,fv = regularize_joint_fisher((total_k/tokens).float(), (total_v/tokens).float())
    args.output.mkdir(parents=True, exist_ok=False)
    fisher_path = args.output/'anchor_fisher.pt'
    torch.save(dict(coordinate_system='F_anchor_calfa_affine_z_v1',scale_contract='calfa_center_radial_v1',
                    coordinate_sha256=identity,model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
                    k_weights=fk,v_weights=fv,windows=len(records),tokens=tokens,
                    input_ids_sha256=sha(args.input_ids),objective='full shifted next-token mean CE',
                    gradient_node='K post-k_norm/pre-RoPE; V v_proj; signed activation gradients cast to FP32 before coordinate transform/square',
                    stored_scale_detached=True,shrinkage=.05,winsor_quantile=.995), fisher_path)
    import os
    assets = {}
    for role, path in [('k_bundle', args.bundle/'codebook.pt'),('v_bundle', args.bundle/'codebook_v.pt'),('anchor_fisher', fisher_path)]:
        assets[role] = dict(path=os.path.relpath(path.resolve(),args.output.resolve()),sha256=sha(path))
    manifest = dict(method='CALFA-KV',asset_ready=True,G=8,bits=8,anchor_capacity=56,
                    model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',assets=assets,
                    coordinate_sha256=identity,windows=len(records),input_ids_sha256=sha(args.input_ids))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Anchor Fisher and final-asset manifest saved:', args.output)


if __name__ == '__main__':
    main()
