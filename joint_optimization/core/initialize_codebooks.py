"""Warm-start both codebooks against projected attention output."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from attention_objective import decode_fixed_indices, reference_state
from kv_objective import kv_output_sse
from direction import group_coordinate_scale, propose_direction
from policy import FIT, VALIDATION, SPLITS, accepts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temp = Path(path).with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)






class JointProblem:
    """Frozen calibration and actual encoder; no query/model/scale learning."""

    def __init__(self, cfg):
        from safetensors import safe_open
        from kvquant.frozen_training import load_layer
        from grouping import load_queries
        from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
        from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
        from calfa_scale import encode_features

        assert os.environ["ARTIFACTS"] == cfg["assets"]
        assert os.environ["SGLANG_VQ_CODEBOOK_PATH"] == str(Path(cfg["assets"]) / "bundle/codebook.pt")
        assert os.environ["SGLANG_VQ_V_CODEBOOK_PATH"] == str(Path(cfg["assets"]) / "bundle/codebook_v.pt")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(20261020)
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method="tcp://[::1]:%d" % cfg["port"], backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=1)
        self.pool = UnifiedInt2HPKVPool(num_quant_pages=256, hp_dtype=torch.bfloat16,
            hp_prefix_tokens=0, hp_recent_tokens=64, dtype="int2", head_num=8, head_dim=128,
            layer_num=36, device="cuda", enable_memory_saver=False, max_req_slots=2,
            model_dtype=torch.bfloat16, kv_cache_quant_group_size=128, scale_dtype=torch.float16)
        self.layer = cfg["layer"]
        assert self.pool._vq.scale_contract == "calfa_fp16_scale_v1"
        data, self.features = load_layer(cfg["features"], self.layer, cfg["input_ids_sha256"])
        self.raw = data["k_feature"].reshape(-1, 8, 128).cuda()
        self.value = data["v_feature"].reshape(-1, 8, 128).cuda()
        del data
        self.weight = self.pool._vq.channel_weights[self.layer].clone()
        self.permutation = self.pool._vq.channel_permutations[self.layer].long().clone()
        self.identity = torch.arange(128, device="cuda").repeat(8, 1)
        self.parent = torch.stack([self.pool._vq.cb16[self.layer],
                                   self.pool._vq_v.cb16[self.layer]]).float().clone()
        assert self.parent.shape == (2, 8, 16, 256, 8)
        self.units = group_coordinate_scale(self.parent)
        self.normalized, self.scale = encode_features(self.raw, self.weight)
        self.queries, self.query_ids = load_queries(cfg, self.layer, self.features)
        with safe_open(cfg["projection_shard"], framework="pt", device="cpu") as f:
            self.projection = f.get_tensor(cfg["projection_key"]).float().cuda()
        assert self.projection.shape == (2560, 4096)
        self.refs = []
        with torch.no_grad():
            for w, (q, cos, sin, positions) in enumerate(self.queries):
                sl = slice(w * 2048, (w + 1) * 2048)
                ref = reference_state(q, self.raw[sl][None], self.value[sl][None],
                                      cos, sin, positions, self.projection)
                assert float(kv_output_sse(self.raw[sl].float(), self.value[sl].float(), ref)[0]) == 0
                self.refs.append(ref)
        self.energy = [float(r["energy"]) for r in self.refs]
        assert len(self.energy) == 64 and all(e > 0 for e in self.energy)
        self.fit_root = math.sqrt(math.fsum(self.energy[w] for w in FIT))

    def assign(self, books):
        from vector_codec import encode
        with torch.no_grad():
            k = torch.stack([encode(self.normalized[:, h], self.permutation[h], books[0, h])
                             for h in range(8)], 1)
            v = torch.stack([encode(self.value[:, h], self.identity[h], books[1, h])
                             for h in range(8)], 1)
        return k, v

    def decode(self, books, labels, w):
        sl = slice(w * 2048, (w + 1) * 2048)
        kd = decode_fixed_indices(books[0], labels[0][sl], self.scale[sl],
                                  self.weight, self.permutation)
        hi = torch.arange(8, device=books.device)[None, :, None]
        gi = torch.arange(16, device=books.device)[None, None, :]
        vd = books[1][hi, gi, labels[1][sl]].reshape(2048, 8, 128)
        return kd, vd

    def metrics(self, sse):
        return {name: math.fsum(sse[w] for w in windows) /
                math.fsum(self.energy[w] for w in windows) for name, windows in SPLITS.items()}

    def evaluate(self, books):
        labels = self.assign(books)
        with torch.no_grad():
            sse = [float(kv_output_sse(*self.decode(books, labels, w), r)[0])
                   for w, r in enumerate(self.refs)]
        scores = self.metrics(sse)
        assert all(math.isfinite(x) for x in scores.values())
        return scores, sse, labels

    def functions(self, labels):
        def one(w):
            def residual(u):
                books = u.reshape_as(self.units) * self.units
                return kv_output_sse(*self.decode(books, labels, w), self.refs[w])[1].reshape(-1) / self.fit_root
            return residual
        return [one(w) for w in FIT]

    def native_check(self, books, labels):
        from sglang.srt.mem_cache import vq_codebook
        from sglang.srt.mem_cache.vq_codebook import vq_unpack_idx
        layer = self.layer
        for side, codec in enumerate([self.pool._vq, self.pool._vq_v]):
            half = books[side].half()
            codec.cb16[layer].copy_(half)
            codec.cb_sq[layer].copy_(.5 * half.cpu().float().square().sum(-1))
        compared = 0
        with torch.no_grad():
            for w in range(64):
                sl = slice(w * 2048, (w + 1) * 2048)
                vq_codebook._VQ_PAD_CACHE.clear()
                self.pool.set_kv_buffer(SimpleNamespace(layer_id=layer, calfa_v_rotation_absorbed=True),
                    torch.arange(2048, device="cuda"), self.raw[sl], self.value[sl])
                k = vq_unpack_idx(self.pool._vq_k_idx_big[layer, :2048], 8).long()
                v = vq_unpack_idx(self.pool._vq_v_idx_big[layer, :2048], 8).long()
                assert torch.equal(k, labels[0][sl]) and torch.equal(v, labels[1][sl]), w
                assert torch.equal(self.scale[sl].view(torch.int16),
                    self.pool._vq_k_sz_big[layer, :2048, 0, 0].view(torch.int16)), w
                assert (self.pool._vq_v_sz_big[layer, :2048, ..., 0] == 1).all(), w
                compared += k.numel() + v.numel()
        return dict(passed=True, windows=64, kv_indices_compared=compared,
                    k_scale_bitwise_equal=True, v_scale_one=True)


def run(cfg, config_path):
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    identity = dict(pid=os.getpid(),
        start_ticks=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        layer=cfg["layer"], method=cfg["method"])
    def status(phase, **kw):
        save(out / "status.json", dict(identity, phase=phase, **kw,
             at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    start = time.monotonic()
    status("loading_frozen_joint_problem")
    problem = JointProblem(cfg)
    current = problem.parent.clone()
    parent_metrics, parent_sse, labels = problem.evaluate(current)
    save(out / "parent_native_check.json", problem.native_check(current, labels))
    torch.save(dict(books_fp16=current.half().cpu(), units=problem.units.cpu(),
                    weights=problem.weight.cpu(), permutations=problem.permutation.cpu(),
                    metrics=parent_metrics, sse=parent_sse, energy=problem.energy), out / "parent.pt")
    metrics, sse = parent_metrics, parent_sse
    history = []
    for iteration in range(cfg["rounds"]):
        status("joint_output_direction", iteration=iteration)
        point = (current / problem.units).reshape(-1)
        assert torch.equal((point.reshape_as(current) * problem.units).half(), current.half())
        direction, gradient, step, diagnostic = propose_direction(
            problem.functions(labels), point, cfg["method"], cfg["cg_iterations"], cfg["cg_tolerance"])
        torch.save(dict(point=point.cpu(), gradient=gradient.cpu(), direction=direction.cpu(),
                        raw_step=step.cpu(), diagnostic=diagnostic),
                   out / ("direction_%02d.pt" % iteration))
        if diagnostic["zero_gradient"]:
            history.append(dict(iteration=iteration, stop="zero_gradient"))
            break
        accepted, trials = False, []
        for b in range(cfg["backtracks"]):
            alpha = cfg["initial_relative_step"] * .5 ** b
            proposal = ((point + alpha * direction).reshape_as(current) * problem.units).half().float()
            assert torch.isfinite(proposal).all()
            status("native_reassignment_acceptance", iteration=iteration, backtrack=b)
            candidate_metrics, candidate_sse, candidate_labels = problem.evaluate(proposal)
            passed, reasons = accepts(metrics, candidate_metrics, parent_metrics)
            file = out / ("proposal_%02d_%02d.pt" % (iteration, b))
            torch.save(dict(books_fp16=proposal.half().cpu(), metrics=candidate_metrics,
                            sse=candidate_sse, energy=problem.energy, alpha=alpha,
                            accepted=passed, reasons=reasons), file)
            trials.append(dict(backtrack=b, alpha=alpha, accepted=passed, reasons=reasons,
                               file=file.name, sha256=sha(file), metrics=candidate_metrics))
            if passed:
                current, metrics, sse, labels = proposal, candidate_metrics, candidate_sse, candidate_labels
                accepted = True
                break
        history.append(dict(iteration=iteration, accepted=accepted, trials=trials, direction=diagnostic))
        save(out / "history.json", history)
        if not accepted:
            break
    status("final_native_verification")
    final_metrics, final_sse, final_labels = problem.evaluate(current)
    assert final_metrics == metrics and final_sse == sse
    assert all(torch.equal(a, b) for a, b in zip(labels, final_labels))
    save(out / "selected_native_check.json", problem.native_check(current, final_labels))
    selected = out / "selected_pc.pt"
    torch.save(dict(books_fp16=current.half().cpu(), units=problem.units.cpu(),
        weights=problem.weight.cpu(), permutations=problem.permutation.cpu(),
        metrics=metrics, sse=sse, energy=problem.energy,
        labels_k=final_labels[0].cpu().to(torch.uint8),
        labels_v=final_labels[1].cpu().to(torch.uint8)), selected)
    save(out / "complete.json", dict(
        passed=True, protocol=cfg["protocol"], method=cfg["method"], layer=cfg["layer"],
        config_sha256=sha(config_path), selected_sha256=sha(selected), parent_sha256=sha(out / "parent.pt"),
        source_sha256=cfg["source_sha256"], feature_identities=problem.features,
        query_identities=problem.query_ids, projection_shard_sha256=sha(cfg["projection_shard"]),
        parent_metrics=parent_metrics, selected_metrics=metrics, history=history,
        accepted_steps=sum(int(h.get("accepted", False)) for h in history),
        permutation_w_scale_unchanged=True, both_k_v_codewords_learned=True,
        kl_used_for_learning_or_acceptance=False, raw_k_mse_used_for_learning_or_acceptance=False,
        independent_holdout_claim=False, accuracy_claim=False,
        duration_seconds=time.monotonic() - start,
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    status("joint_output_learning_complete")
    torch.distributed.destroy_process_group()
