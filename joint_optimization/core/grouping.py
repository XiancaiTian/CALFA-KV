"""CALFA group/codebook refinement calibration-only search; preserve each evaluated P and its exact C."""
import datetime
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import torch

from vector_codec import decode, encode, fit_two_groups
from grouping_metrics import Moments, attention_diagnostics, codeword_summary
from grouping_rules import acceptance, nominate, self_checks, validate_permutation
from kvquant.frozen_training import load_layer
from artifact_io import sha, save
from sglang.srt.distributed import (
    init_distributed_environment, initialize_model_parallel,
)
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
from sglang.srt.mem_cache.vq_codebook import vq_unpack_idx
from calfa_scale import encode_features


def read(path):
    return json.loads(Path(path).read_text())


def status(out, **fields):
    start = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    save(out / "status.json", dict(pid=os.getpid(), start_ticks=start,
         at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields))


def load_queries(config, layer, feature_ids):
    queries, identities = [], []
    for window in range(64):
        base = Path(config["q_data"]) / ("window_%03d" % window)
        meta = read(base / "complete.json")
        record = meta["layers"][layer]
        assert meta["input_ids_sha256"] == config["input_ids_sha256"]
        assert record["layer"] == layer and record["kv_exact"]
        assert record["original_kv_sha256"] == feature_ids[window]["file_sha256"]
        assert sha(base / record["file"]) == record["sha256"]
        assert sha(base / "rotary.pt") == meta["rotary_sha256"]
        q = torch.load(base / record["file"], map_location="cuda", weights_only=True)
        rotary = torch.load(base / "rotary.pt", map_location="cuda", weights_only=True)
        assert q["query_positions"] == list(range(127, 2048, 128))
        queries.append((q["q_pre_rope"], rotary["cos"], rotary["sin"],
                        q["query_positions"]))
        identities.append(dict(window=window, q_sha256=record["sha256"],
                               rotary_sha256=meta["rotary_sha256"]))
    return queries, identities


def measure(head, permutation, books, raw, value, fisher, normalized,
            scale, weight, queries):
    decoded, labels = decode(normalized[:, head], scale, weight[head],
                             permutation, books)
    residual = (decoded - raw[:, head].float()).reshape(64, 2048, 128)
    history = residual[:, :1984]
    importance = fisher[:, head].reshape(64, 2048, 128)[:, :1984]
    attention = []
    for window, (q, cos, sin, positions) in enumerate(queries):
        sl = slice(window * 2048, (window + 1) * 2048)
        row = attention_diagnostics(
            q[:, :, head * 4:(head + 1) * 4],
            raw[sl, head:head + 1][None],
            decoded[sl, None][None],
            value[sl, head:head + 1][None], cos, sin, positions)
        attention.append({k: row[k].cpu() for k in
                          ("kl", "quadratic_kl", "output_sse", "output_signal")})
    tensors = {k: torch.stack([r[k] for r in attention]) for k in attention[0]}
    kl = tensors["kl"]
    error = history.double().square()
    moments = Moments(1, 128, "cuda")
    moments.add(history.reshape(-1, 1, 128))
    grouped = error[..., permutation].reshape(-1, 16, 8).mean(-1)
    counts = torch.zeros(16, 256, device="cuda", dtype=torch.int64)
    lab = labels.reshape(64, 2048, 16)[:, :1984].reshape(-1, 16)
    counts.scatter_add_(1, lab.T, torch.ones_like(lab.T))
    metrics = dict(kl=float(kl.mean()), gpqa_kl=float(kl[:48].mean()),
                   code_kl=float(kl[48:].mean()),
                   output_relative_sse=float(tensors["output_sse"].sum()
                                             / tensors["output_signal"].sum()),
                   mse=float(error.mean()),
                   fisher_mse=float((error * importance.double()).sum()
                                    / importance.double().sum()),
                   group_mse=grouped.mean(0).cpu().tolist(),
                   group_mse_p95=torch.quantile(grouped, .95, dim=0).cpu().tolist(),
                   group_mse_p99=torch.quantile(grouped, .99, dim=0).cpu().tolist(),
                   codewords=codeword_summary(counts[None].cpu()))
    tensors.update(residual=moments.finish(), occupancy=counts.cpu())
    return metrics, tensors


def save_state(path, permutation, books, metrics, tensors, **metadata):
    torch.save(dict(permutation=permutation.cpu(), books_fp32=books.float().cpu(),
                    books_fp16=books.half().cpu(), metrics=metrics,
                    tensors=tensors, **metadata), path)
    return dict(file=path.name, sha256=sha(path), metrics=metrics, **metadata)


def check_parent_native(pool, layer, raw, value, normalized, scale,
                        permutations, books):
    """All64 real pool index/scale equality plus same-pointer C mutation."""
    labels = torch.stack([
        encode(normalized[:, head], permutations[head], books[head])
        for head in range(8)], 1)
    for window in range(64):
        sl = slice(window * 2048, (window + 1) * 2048)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer, calfa_v_rotation_absorbed=True),
            torch.arange(2048, device="cuda"), raw[sl], value[sl])
        expected = vq_unpack_idx(pool._vq_k_idx_big[layer, :2048], 8).long()
        assert torch.equal(scale[sl].view(torch.int16),
                           pool._vq_k_sz_big[layer, :2048, 0, 0].view(torch.int16))
        assert torch.equal(labels[sl], expected), ("native_assignment", window)
    # Mutating one FP16 buffer to reverse codeword order must not reuse padding.
    cb = books[0].half().clone()
    before = encode(normalized[:2048, 0], permutations[0], cb)
    address = cb.data_ptr()
    cb.copy_(cb.flip(1))
    assert address == cb.data_ptr()
    after = encode(normalized[:2048, 0], permutations[0], cb)
    fresh = encode(normalized[:2048, 0], permutations[0], cb.clone())
    assert torch.equal(after, fresh) and (after != before).any()
    return dict(passed=True, all_eight_heads_match_actual_pool=True,
                all_64_windows_indices_match_actual_pool=True,
                codebook_squared_norm_uses_native_cpu_reduction=True,
                scale_bitwise_equal=True, same_pointer_mutation_matches_fresh=True)


def run(config, out, config_path, admission_only=False):
    assert config["rounds"] == 2 and config["candidate_count"] == 6
    assert config["layer"] in range(36)
    assert config["independent_holdout_claim"] is False
    assert config["protocol"] == "calfa_group_codebook_refinement_v1"
    for file, expected in config["input_sha256"].items():
        assert sha(file) == expected, file
    save(out / "rule_checks.json", self_checks())
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:%d" % config["port"], backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    pool = UnifiedInt2HPKVPool(num_quant_pages=256, hp_dtype=torch.bfloat16,
        hp_prefix_tokens=0, hp_recent_tokens=64, dtype="int2", head_num=8,
        head_dim=128, layer_num=36, device="cuda", enable_memory_saver=False,
        max_req_slots=16, model_dtype=torch.bfloat16,
        kv_cache_quant_group_size=128, scale_dtype=torch.float16)
    assert pool._vq.scale_contract == "calfa_fp16_scale_v1"
    layer = config["layer"]
    status(out, phase="loading_frozen_features", layer=layer)
    data, feature_ids = load_layer(config["features"], layer,
                                   config["input_ids_sha256"])
    raw = data["k_feature"].reshape(-1, 8, 128).cuda()
    value = data["v_feature"].reshape(-1, 8, 128).cuda()
    fisher = data["k_grad_sq_fp32"].reshape(-1, 8, 128).cuda()
    del data
    weight = pool._vq.channel_weights[layer].clone()
    permutations = pool._vq.channel_permutations[layer].long().clone()
    with Path(config["assets"], "quantizers.pkl").open("rb") as stream:
        quantizers = pickle.load(stream)
    books = torch.as_tensor(
        quantizers["model.layers.%d.self_attn.k_proj" % layer],
        device="cuda").reshape(8, 16, 256, 8).float().clone()
    del quantizers
    assert torch.equal(books.half(), pool._vq.cb16[layer])
    normalized, scale = encode_features(raw, weight)
    queries, query_ids = load_queries(config, layer, feature_ids)
    proof = check_parent_native(pool, layer, raw, value, normalized, scale,
                                permutations, books)
    parent = torch.load(config["parent_layer_tensors"], weights_only=True,
                        map_location="cpu")
    assert torch.equal(permutations.cpu(), parent["permutation"])
    current_metrics, current_tensors = [], []
    for head in range(8):
        status(out, phase="parent_admission", layer=layer, head=head)
        metrics, tensors = measure(head, permutations[head], books[head], raw,
                                   value, fisher, normalized, scale, weight, queries)
        expected = parent["attention"]["kl"][:, :, head * 4:(head + 1) * 4]
        torch.testing.assert_close(tensors["kl"], expected, atol=1e-6, rtol=2e-4)
        torch.testing.assert_close(tensors["output_sse"],
            parent["attention"]["output_sse"][:, :, head * 4:(head + 1) * 4],
            atol=5e-6, rtol=4e-4)
        current_metrics.append(metrics)
        current_tensors.append(tensors)
        save_state(out / ("parent_head_%d.pt" % head), permutations[head],
                   books[head], metrics, tensors, head=head, kind="parent")
    proof.update(all_64_windows_parent_local_kl_verified=True,
                 config_sha256=sha(config_path), source_sha256=config['source_sha256'])
    save(out / "parent_admission.json", proof)
    save(out / "input_identities.json", dict(features=feature_ids, queries=query_ids))
    if admission_only:
        status(out, phase="parent_admission_passed", layer=layer)
        torch.distributed.destroy_process_group()
        return
    selections = []
    for round_id in range(2):
        for head in range(8):
            head_dir = out / ("round_%d_head_%d" % (round_id, head))
            head_dir.mkdir()
            old_p, old_c = permutations[head].clone(), books[head].clone()
            old_m, old_t = current_metrics[head], current_tensors[head]
            seed = 20261020 + layer * 1000 + head * 10 + round_id
            proposed = nominate(old_p.cpu().numpy(),
                parent["source"]["correlation"][head].numpy(),
                old_t["residual"]["correlation"][0].numpy(), seed)
            save(head_dir / "proposals.json", dict(seed=seed, proposals=proposed))
            controls, candidates = {}, []
            for candidate_id, proposal in enumerate(proposed):
                groups = tuple(proposal["move"][:2])
                status(out, phase="fitting_matched_control", layer=layer,
                       round=round_id, head=head, candidate=candidate_id)
                if groups not in controls:
                    fitted, diagnostic = fit_two_groups(
                        normalized[:, head], fisher[:, head], weight[head],
                        scale, old_p, groups, seed)
                    control_c = old_c.clone()
                    control_c[list(groups)] = fitted
                    cm, ct = measure(head, old_p, control_c, raw, value, fisher,
                                      normalized, scale, weight, queries)
                    file = head_dir / ("control_%02d_%02d.pt" % groups)
                    record = save_state(file, old_p, control_c, cm, ct, head=head,
                        round=round_id, kind="no_swap_retrain", groups=list(groups),
                        seed=seed, training_diagnostics=diagnostic)
                    controls[groups] = record
                status(out, phase="fitting_swap_candidate", layer=layer,
                       round=round_id, head=head, candidate=candidate_id)
                from grouping_rules import swap_pairs
                proposed_p = torch.as_tensor(swap_pairs(
                    old_p.cpu().numpy(), proposal["move"]), device="cuda")
                fitted, diagnostic = fit_two_groups(
                    normalized[:, head], fisher[:, head], weight[head],
                    scale, proposed_p, groups, seed)
                proposed_c = old_c.clone()
                proposed_c[list(groups)] = fitted
                cm, ct = measure(head, proposed_p, proposed_c, raw, value, fisher,
                                  normalized, scale, weight, queries)
                decision = acceptance(old_m, controls[groups]["metrics"], cm)
                record = save_state(head_dir / ("candidate_%d.pt" % candidate_id),
                    proposed_p, proposed_c, cm, ct, head=head, round=round_id,
                    kind="swap_and_retrain", proposal=proposal, seed=seed,
                    training_diagnostics=diagnostic,
                    matched_control=controls[groups]["file"], decision=decision)
                candidates.append(record)
                save(head_dir / "candidates.json", candidates)
            eligible = [r for r in candidates if r["decision"]["accepted"]]
            winner = min(eligible, key=lambda r: (r["metrics"]["kl"], r["file"])) if eligible else None
            if winner:
                selected = torch.load(head_dir / winner["file"],
                                      weights_only=False, map_location="cpu")
                permutations[head] = selected["permutation"].cuda()
                books[head] = selected["books_fp32"].cuda()
                current_metrics[head], current_tensors[head] = (
                    selected["metrics"], selected["tensors"])
            selected_record = dict(round=round_id, head=head,
                accepted=winner is not None, selected_file=winner["file"] if winner else None,
                parent_metrics=old_m, selected_metrics=current_metrics[head],
                candidates=candidates, controls=list(controls.values()))
            save(head_dir / "selection.json", selected_record)
            selections.append(selected_record)
            save_state(head_dir / "selected.pt", permutations[head], books[head],
                       current_metrics[head], current_tensors[head],
                       head=head, round=round_id, kind="selected_exact_pc")
    for head in range(8):
        validate_permutation(permutations[head].cpu().numpy())
    torch.save(dict(layer=layer, permutations=permutations.cpu(),
                    books_fp32=books.cpu(), books_fp16=books.half().cpu()),
               out / "selected_pc.pt")
    save(out / "complete.json", dict(complete=True, layer=layer,
        config_sha256=sha(config_path), selected_sha256=sha(out / "selected_pc.pt"),
        accepted_moves=sum(r["accepted"] for r in selections),
        selected_metrics=current_metrics, rounds=2, local_calibration_only=True,
        independent_holdout_claim=False, not_final_vocabulary_kl=True,
        independently_verified=False))
    status(out, phase="search_complete_awaiting_independent_verification", layer=layer)
    torch.distributed.destroy_process_group()
