"""Bounded CALFA-KV Anchor storage, independent of codec and flush policy.

Only callers holding *true pre-quantization* BF16 K/V may admit candidates.
K is post-k_norm/pre-RoPE in canonical channel order; stored Anchors are FP16.
The caller supplies the actual jointly weighted reconstruction score and the
native quantized slots/absolute positions. This class never infers eviction
from a nominal Recent window, never modifies req_to_token, and never keeps a
full-history high-precision sidecar.

Selection is lexicographic: decreasing score, then increasing logical
position. Every request/layer/head has its own 56 entries; K/V share the
same selected positions. All data-dependent operations stay on the device,
including invariant assertions, so fixed-shape calls can be CUDA-captured.
This component alone is NOT numerical acceptance of the attention path.
"""

import torch


CAPACITY = 56
KV_HEADS = 8
HEAD_DIM = 128
INVALID_POSITION = torch.iinfo(torch.int64).max


def _require(condition, message):
    """Fail closed without a graph-breaking GPU scalar read.

    An invalid GPU input is a device assertion and invalidates that worker;
    it is not silently dropped to continue an allegedly valid experiment.
    """
    torch._assert_async(condition, message)


class AnchorCache:
    """Request-indexed bounded FP16 storage; enable only for TP1 CALFA-KV affine g8/bits8."""

    def __init__(self, layer_count, request_capacity, device):
        if layer_count <= 0 or request_capacity <= 0:
            raise ValueError("Layer and request capacities must be positive")
        self.layer_count = int(layer_count)
        self.request_capacity = int(request_capacity)
        self.device = torch.device(device)
        shape = (layer_count, request_capacity, KV_HEADS, CAPACITY)
        self.positions = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.quant_slots = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.scores = torch.full(shape, -torch.inf, device=device)
        self.k = torch.empty((*shape, HEAD_DIM), dtype=torch.float16, device=device)
        self.v = torch.empty_like(self.k)
        self.counts = torch.zeros(shape[:-1], dtype=torch.int32, device=device)
        self.generation = torch.zeros(request_capacity, dtype=torch.int64, device=device)

    def _check_requests(self, requests):
        if requests.dtype != torch.int64 or requests.ndim != 1:
            raise ValueError("Request indices must be an int64 vector")
        if requests.device != self.positions.device:
            raise ValueError("Request indices must share the cache device")
        _require(((requests >= 0) & (requests < self.request_capacity)).all(),
                 "Anchor request index outside allocated pool")
        unique = (requests[:, None] == requests[None, :]).sum(-1) == 1
        _require(unique.all(), "Duplicate requests in one Anchor operation")

    def _check_layer(self, layer):
        if not 0 <= layer < self.layer_count:
            raise ValueError("Anchor layer outside allocated pool")

    @torch.no_grad()
    def reset(self, requests):
        """Release/reuse a request slot across ALL layers before reuse.

        Old K/V bytes need not be zeroed: invalid positions and slots make them
        unreachable. No asynchronous host-side bookkeeping is used as truth.
        """
        self._check_requests(requests)
        self.positions.index_fill_(1, requests, -1)
        self.quant_slots.index_fill_(1, requests, -1)
        self.scores.index_fill_(1, requests, -torch.inf)
        self.counts.index_fill_(1, requests, 0)
        self.generation.index_add_(0, requests, torch.ones_like(requests))

    @torch.no_grad()
    def admit(self, layer, requests, positions, quant_slots, valid, raw_k, raw_v,
              scores):
        """Merge actual native migrants with the previous bounded directory.

        requests: [B], int64 request-pool IDs, unique within the call.
        positions/quant_slots/valid: [B,T], absolute int64 IDs and bool mask.
        raw_k/raw_v: [B,T,8,128], true BF16 values BEFORE VQ encoding.
        scores: [B,T,8], nonnegative finite FP32 joint K/V error.

        Works for native page-8 flushes and eager prefill quantized regions.
        Invalid/padding entries never participate. Candidate positions may be
        unordered, but repeated valid positions or slot aliases are errors.
        The caller must exclude actual Prefix/Recent positions, not merely
        subtract 64 from a sequence length. P0 permits positions starting at zero; the caller must exclude
        the actual native Recent positions.
        """
        self._check_layer(layer)
        self._check_requests(requests)
        if positions.ndim != 2 or positions.shape[0] != requests.numel():
            raise ValueError("Candidate metadata must be [batch,tokens]")
        batch, tokens = positions.shape
        metadata_shape = (batch, tokens)
        for value in (positions, quant_slots):
            if value.dtype != torch.int64 or value.shape != metadata_shape:
                raise ValueError("Position and quantized slot IDs must be int64 [B,T]")
        if valid.dtype != torch.bool or valid.shape != metadata_shape:
            raise ValueError("Candidate validity must be bool [B,T]")
        for value in (raw_k, raw_v):
            if value.dtype != torch.bfloat16 or value.shape != (*metadata_shape, 8, 128):
                raise ValueError("Candidates must be true BF16 [B,T,8,128] K/V")
        if scores.dtype != torch.float32 or scores.shape != (*metadata_shape, 8):
            raise ValueError("Scores must be FP32 [B,T,8]")
        for value in (positions, quant_slots, valid, raw_k, raw_v, scores):
            if value.device != self.positions.device:
                raise ValueError("All admission tensors must share the cache device")
        if not batch or not tokens:
            return
        _require((~valid | ((positions >= 0) & (quant_slots >= 0))).all(),
                 "Anchor candidate has an invalid position or quantized slot")
        valid_h = valid[:, None, :].expand(batch, KV_HEADS, tokens)
        new_scores = scores.transpose(1, 2)
        _require((~valid_h | (torch.isfinite(new_scores) & (new_scores >= 0))).all(),
                 "Nonfinite or negative Anchor score")
        # Check the actual FP16 storage cast, including BF16 -> FP16 overflow.
        new_k = raw_k.permute(0, 2, 1, 3).to(torch.float16)
        new_v = raw_v.permute(0, 2, 1, 3).to(torch.float16)
        _require((~valid_h | (torch.isfinite(new_k).all(-1)
                             & torch.isfinite(new_v).all(-1))).all(),
                 "Invalid true K/V or FP16 Anchor overflow")

        old_pos = self.positions[layer].index_select(0, requests)
        old_slots = self.quant_slots[layer].index_select(0, requests)
        old_scores = self.scores[layer].index_select(0, requests)
        old_valid = old_pos >= 0
        all_valid = torch.cat((old_valid, valid_h), -1)
        all_pos = torch.cat((old_pos, positions[:, None, :].expand_as(valid_h)), -1)
        all_slots = torch.cat((old_slots, quant_slots[:, None, :].expand_as(valid_h)), -1)
        all_scores = torch.cat((old_scores, new_scores), -1).masked_fill(~all_valid, -torch.inf)

        # Position ordering is the secondary key. Stable score ordering then
        # retains the earlier logical position on every exact score tie.
        ordered = all_pos.masked_fill(~all_valid, INVALID_POSITION).argsort(dim=-1, stable=True)
        ordered_pos = all_pos.gather(-1, ordered)
        ordered_valid = all_valid.gather(-1, ordered)
        repeated = ((ordered_pos[..., 1:] == ordered_pos[..., :-1])
                    & ordered_valid[..., 1:] & ordered_valid[..., :-1])
        _require((~repeated).all(), "Duplicate Anchor logical position")
        sorted_slots = all_slots.masked_fill(~all_valid, INVALID_POSITION).sort(-1).values
        aliased = ((sorted_slots[..., 1:] == sorted_slots[..., :-1])
                   & (sorted_slots[..., 1:] != INVALID_POSITION))
        _require((~aliased).all(), "Quantized slot reused before Anchor release")
        rank = all_scores.gather(-1, ordered).argsort(dim=-1, descending=True, stable=True)
        selected = ordered.gather(-1, rank[..., :CAPACITY])
        chosen_valid = all_valid.gather(-1, selected)

        # Gather at most A rows from each source. No persistent raw history or
        # concatenated [A+T,128] exact-value buffer is added to the cache.
        old_index = selected.clamp(max=CAPACITY - 1)[..., None].expand(-1, -1, -1, HEAD_DIM)
        new_index = (selected - CAPACITY).clamp(0, tokens - 1)[..., None].expand_as(old_index)
        is_new = (selected >= CAPACITY)[..., None]
        for destination, incoming in ((self.k, new_k), (self.v, new_v)):
            old = destination[layer].index_select(0, requests).gather(2, old_index)
            new = incoming.gather(2, new_index)
            result = torch.where(is_new, new, old)
            result = result.masked_fill(~chosen_valid[..., None], 0)
            destination[layer].index_copy_(0, requests, result)
        self.positions[layer].index_copy_(
            0, requests, all_pos.gather(-1, selected).masked_fill(~chosen_valid, -1))
        self.quant_slots[layer].index_copy_(
            0, requests, all_slots.gather(-1, selected).masked_fill(~chosen_valid, -1))
        self.scores[layer].index_copy_(
            0, requests, all_scores.gather(-1, selected).masked_fill(~chosen_valid, -torch.inf))
        self.counts[layer].index_copy_(0, requests, chosen_valid.sum(-1).to(torch.int32))

    @torch.no_grad()
    def lookup(self, layer, requests, positions, quant_slots):
        """Resolve request+position+slot, for dense-prefix overlay/diagnostics.

        Returns hit [B,T,8] and canonical FP16 K/V [B,T,8,128]. Invalid or
        stale positions/slots cannot expose old request bytes. The attention
        decode kernel should consume the bounded arrays directly, avoiding
        materialization of a full-history high-precision sidecar.
        """
        self._check_layer(layer)
        self._check_requests(requests)
        if positions.ndim != 2 or positions.shape[0] != requests.numel():
            raise ValueError("Lookup positions must be [B,T]")
        for value in (positions, quant_slots):
            if value.dtype != torch.int64 or value.shape != positions.shape:
                raise ValueError("Lookup position/slot IDs must be int64 [B,T]")
            if value.device != self.positions.device:
                raise ValueError("Lookup metadata must share the cache device")
        pos = self.positions[layer].index_select(0, requests)
        slots = self.quant_slots[layer].index_select(0, requests)
        matches = ((pos[:, :, None, :] == positions[:, None, :, None])
                   & (slots[:, :, None, :] == quant_slots[:, None, :, None])
                   & (pos[:, :, None, :] >= 0))
        hit = matches.any(-1)
        index = matches.to(torch.int32).argmax(-1)[..., None].expand(-1, -1, -1, HEAD_DIM)
        values = []
        for source in (self.k, self.v):
            rows = source[layer].index_select(0, requests).gather(2, index)
            values.append(rows.masked_fill(~hit[..., None], 0).transpose(1, 2))
        return hit.transpose(1, 2), *values

    def storage_bytes(self):
        """Persistent allocation only; admission/prefill temporaries are separate."""
        return {name: value.numel() * value.element_size()
                for name, value in vars(self).items() if isinstance(value, torch.Tensor)}
