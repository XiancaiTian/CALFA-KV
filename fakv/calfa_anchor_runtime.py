"""Anchor56 over frozen CALFA affine G8/256 assets; native page-8 lifecycle."""
import hashlib
import json
import os
from pathlib import Path
import torch
from calfa_anchor_cache import AnchorCache
from calfa_anchor_codec import reconstruct_candidates, score_actual_reconstruction

MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
COORDINATE = "F_anchor_calfa_affine_z_v1"
SCALE_CONTRACT = "calfa_center_radial_v1"

def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()

def coordinate_tensors(vq):
    freq = vq.freq_idx
    permutation = torch.stack((freq, freq + 64), -1).flatten(-2).contiguous()
    wanted = torch.arange(128, device=permutation.device).expand_as(permutation)
    if not torch.equal(permutation.sort(-1).values, wanted):
        raise ValueError("Official RoPE pair map is not a full permutation")
    weights = torch.zeros_like(vq.w_perm).scatter(-1, permutation, vq.w_perm).contiguous()
    if not (torch.isfinite(weights) & (weights > 0)).all():
        raise ValueError("Invalid CALFA channel weights")
    from calfa_anchor_coordinates import coordinate_identity
    digest = coordinate_identity(weights, permutation, vq.affine_center, vq.affine_radial)
    return weights, permutation, digest

class CALFAAnchorRuntime:
    def __init__(self, pool, k_path, v_path, fisher_path):
        actual = (pool.layer_num, pool.head_num, pool.head_dim, pool.v_head_dim,
                  pool.start_layer, pool.hp_prefix_tokens, pool.hp_recent_tokens,
                  pool.N_Q, pool.flush_interval)
        if actual != (36, 8, 128, 128, 0, 0, 64, 8, 8):
            raise ValueError(f"Expected TP1/P0/W64/native page8: {actual}")
        if (pool.hp_dtype != torch.bfloat16 or pool.scale_dtype != torch.float16
                or pool.shadow_enabled or not pool._R_k_identity or not pool._R_v_identity):
            raise ValueError("BF16 Recent, FP16 scales, identity rotations, and no shadow are required")
        if not pool._vq.perm_rope or not pool._vq_v.identity_map:
            raise ValueError("Official W/P K and identity V maps required")
        for vq in (pool._vq, pool._vq_v):
            if (vq.num_groups, vq.codebook_size, vq.group_dim) != (16, 256, 8):
                raise ValueError("Expected affine g8/bits8 K and V")
            if vq.cb16.shape != (36, 8, 16, 256, 8) or vq.cb16.dtype != torch.float16:
                raise ValueError("Invalid live codebook shape/dtype")
            if not torch.equal(vq.mean, torch.zeros_like(vq.mean)):
                raise ValueError("Official zero coding means required")
        if pool._vq_v.pertoken_norm or not pool._vq.pool_heads_scale:
            raise ValueError("Official shared K scale and unscaled V required")
        # CALFA retains the trained FP16 encoding table; do not re-fold or resnap it.
        if pool._vq.scale_contract != SCALE_CONTRACT or pool._vq.affine_center is None or pool._vq.affine_radial is None:
            raise ValueError("CALFA requires final center/radial affine assets")
        path = os.environ.get("SGLANG_CALFA_ANCHOR_ASSET_MANIFEST")
        if not path:
            raise ValueError("A verified actual-asset manifest is required")
        manifest = json.loads(Path(path).read_text())
        if not (manifest["asset_ready"] and manifest["method"] == "CALFA-KV"
                and manifest["model_revision"] == MODEL_REVISION
                and (manifest["G"], manifest["bits"], manifest["anchor_capacity"]) == (8, 8, 56)):
            raise ValueError("Wrong baseline identity in Anchor asset manifest")
        self.identities = {}
        for role, file in (("k_bundle", k_path), ("v_bundle", v_path),
                           ("anchor_fisher", fisher_path)):
            item = manifest["assets"][role]
            if not file or Path(file).resolve() != (Path(path).resolve().parent / item["path"]).resolve():
                raise ValueError(f"Asset path differs for {role}")
            actual_hash = sha256(file)
            if actual_hash != item["sha256"]:
                raise ValueError(f"Asset hash differs for {role}")
            self.identities[str(file)] = actual_hash
        self.weights, self.permutation, identity = coordinate_tensors(pool._vq)
        fisher = torch.load(fisher_path, map_location="cpu", weights_only=False)
        if (fisher["coordinate_system"] != COORDINATE
                or fisher["scale_contract"] != SCALE_CONTRACT
                or fisher["coordinate_sha256"] != identity
                or fisher["model_revision"] != MODEL_REVISION):
            raise ValueError("Anchor Fisher does not match final CALFA D/P/center/radial coordinates")
        for name in ("k_weights", "v_weights"):
            value = fisher[name]
            if value.shape != (36, 8, 128) or value.dtype != torch.float32:
                raise ValueError("Invalid Fisher shape/dtype")
            if not (torch.isfinite(value) & (value > 0)).all():
                raise ValueError("Invalid Fisher values")
        self.fisher_k = fisher["k_weights"].to(pool.device).contiguous()
        self.fisher_v = fisher["v_weights"].to(pool.device).contiguous()
        self.cache = AnchorCache(36, pool.max_req_slots, pool.device)
        # Only the existing Recent arena plus an unread invalid-row sink.
        # This truth buffer never supplies attention directly or retains quantized history.
        # CALFA already owns this bounded Recent truth arena. Borrow it.
        self.raw_k = pool._calfa_hp_raw
        self.center = pool._vq.affine_center
        self.cos_sin = [None] * 36
        self.inv_freq = [None] * 36
        self.audit = None

    def bind_rope(self, layer, cos_sin):
        """Bind actual model cache before capture; never invent a RoPE table."""
        if cos_sin.dtype != torch.float32 or cos_sin.ndim != 2 or cos_sin.shape[1] != 128:
            raise ValueError("Anchor requires actual native FP32 NeoX RoPE cache")
        old = self.cos_sin[layer]
        if old is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Bind Anchor RoPE during eager warmup before CUDA capture")
            self.cos_sin[layer] = cos_sin
            row = cos_sin[1]
            self.inv_freq[layer] = torch.atan2(row[64:], row[:64]).contiguous()
        elif old.data_ptr() != cos_sin.data_ptr() or old.shape != cos_sin.shape:
            raise RuntimeError("Actual RoPE cache changed after Anchor binding")

    def _admit(self, pool, layer, requests, positions, slots, valid, raw_k, raw_v):
        if self.cos_sin[layer] is None:
            raise RuntimeError("Actual model RoPE must be bound before Anchor admission")
        batch, tokens = positions.shape
        flat_pos = positions.reshape(-1).to(torch.int64).contiguous()
        flat_slots = slots.reshape(-1).to(torch.int64).contiguous()
        flat_valid = valid.reshape(-1).bool().contiguous()
        k = raw_k.reshape(-1, 8, 128).contiguous()
        v = raw_v.reshape(-1, 8, 128).contiguous()
        decoded_k, decoded_v = reconstruct_candidates(
            flat_pos, flat_slots, flat_valid, pool.k_buffer[layer], pool.v_buffer[layer],
            (pool._vq.cb16_dec if pool._vq.cb16_dec is not None else pool._vq.cb16)[layer], pool._vq_v.cb16[layer], pool.k_scales_zeros[layer],
            pool._vq.freq_idx[layer], pool._vq.w_perm[layer], self.inv_freq[layer],
            prefix=0, w_folded=pool._vq.cb16_dec is not None,
            pair_major=pool._vq.cb16_dec is not None and pool._vq.cb16_dec_pair_major, center=self.center[layer])
        # Invalid HP/padding candidates route to the existing quant trash row.
        safe_slots = torch.where(flat_valid, flat_slots, pool.quant_size)
        scales = pool.k_scales_zeros[layer].index_select(0, safe_slots)[:, 0, 0].contiguous()
        scores = score_actual_reconstruction(
            k, v, decoded_k, decoded_v, flat_pos, flat_valid, scales,
            self.weights[layer], self.permutation[layer],
            self.fisher_k[layer], self.fisher_v[layer], self.cos_sin[layer])
        self.cache.admit(layer, requests.to(torch.int64), positions.to(torch.int64),
                         slots.to(torch.int64), valid.bool(),
                         k.reshape(batch, tokens, 8, 128), v.reshape(batch, tokens, 8, 128),
                         scores.reshape(batch, tokens, 8))

    def admit_prefill(self, pool, layer, batch, positions, raw_k, raw_v):
        """Called AFTER current-block attention and actual native cache writes."""
        start = 0
        for b, length in enumerate(batch.extend_seq_lens_cpu):
            stop = start + length
            slots = batch.out_cache_loc[start:stop].to(torch.int64)
            valid = (slots >= 0) & (slots < pool.quant_size)
            self._admit(pool, layer, batch.req_pool_indices[b:b + 1],
                        positions[start:stop].view(1, -1), slots.view(1, -1), valid.view(1, -1),
                        raw_k[start:stop].unsqueeze(0), raw_v[start:stop].unsqueeze(0))
            start = stop
        if start != raw_k.shape[0]:
            raise ValueError("Eager prefill token layout differs from native batch metadata")

    def admit_flush(self, pool, plan, requests):
        """Consume original page-8 plan AFTER both native VQ encoders finish."""
        if plan.flush_interval != 8:
            raise ValueError("Anchor cannot change native page-8 flush interval")
        positions = plan.flush_pos.to(torch.int64).view(plan.bs, 8)
        slots = plan.dst_quant_slots.to(torch.int64).view(plan.bs, 8)
        valid = plan.valid_mask.bool().view(plan.bs, 8)
        src = plan.src_hp_slot.clamp_min(0).to(torch.int64)
        for layer in range(pool.layer_num):
            # These native HP arrays still hold pre-quantization truth. The
            # allocator's earlier metadata free does not overwrite these bytes.
            raw_k = self.raw_k[layer].index_select(0, src).view(plan.bs, 8, 8, 128)
            raw_v = pool.hp_v_buffer[layer].index_select(0, src).view(plan.bs, 8, 8, 128)
            self._admit(pool, layer, requests, positions, slots, valid, raw_k, raw_v)

    def overlay_prefix(self, layer, requests, indptr, slots, prefix_k, prefix_v):
        """Overlay canonical pre-RoPE Anchors before native prefix RoPE/map.

        Eager extend only: CPU offsets are allowed because piecewise graph is
        disabled by the frozen protocol. The existing prefix tensors already
        materialize history; no persistent exact-value sidecar is introduced.
        """
        offsets = indptr.to("cpu").tolist()
        for b in range(len(offsets) - 1):
            start, stop = offsets[b:b + 2]
            if stop == start:
                continue
            positions = torch.arange(stop - start, device=slots.device, dtype=torch.int64)[None]
            hit, k, v = self.cache.lookup(layer, requests[b:b + 1].to(torch.int64),
                                          positions, slots[start:stop].to(torch.int64)[None])
            prefix_k[start:stop] = torch.where(hit[0, :, :, None], k[0].to(prefix_k.dtype), prefix_k[start:stop])
            prefix_v[start:stop] = torch.where(hit[0, :, :, None], v[0].to(prefix_v.dtype), prefix_v[start:stop])
        return prefix_k, prefix_v

    def remap_quant_slots(self, old_slots, new_slots):
        if old_slots.numel() == 0:
            return
        source, order = old_slots.to(torch.int64).sort()
        targets = new_slots.to(torch.int64)[order]
        old = self.cache.quant_slots
        index = torch.searchsorted(source, old).clamp_max(source.numel() - 1)
        self.cache.quant_slots.copy_(torch.where((old >= 0) & (source[index] == old), targets[index], old))

    def decode_arguments(self, layer, requests):
        return dict(anchor_k=self.cache.k[layer], anchor_v=self.cache.v[layer],
                    anchor_positions=self.cache.positions[layer], anchor_slots=self.cache.quant_slots[layer],
                    request_indices=requests.to(torch.int64).contiguous())

    def storage_bytes(self):
        result = self.cache.storage_bytes()
        result["fisher_k"] = self.fisher_k.numel() * self.fisher_k.element_size()
        result["fisher_v"] = self.fisher_v.numel() * self.fisher_v.element_size()
        # RoPE is a borrowed model buffer. InvFreq is owned once per bound layer.
        result["inv_freq"] = sum(x.numel() * x.element_size() for x in self.inv_freq if x is not None)
        # raw_k and center are borrowed; pool/codec account for their storage.
        result["coordinate_weights"] = self.weights.numel() * self.weights.element_size()
        result["coordinate_permutation"] = self.permutation.numel() * self.permutation.element_size()
        return result
