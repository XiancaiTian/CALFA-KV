"""Qwen3 Fisher at the actual post-k_norm/pre-RoPE graph node.

No replacement gradient estimator: the objective remains shifted causal CE over
the entire window, labels=input_ids. Signed FP32 gradients are saved BEFORE
squaring. W/P/scale compensation belongs to codebook training, not this module.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import torch


class Qwen3KVGraphCapture:
    def __init__(self, model):
        self.model = model
        self.values = {}
        self.handles = []
        for layer_id, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if not hasattr(attn, "k_norm"):
                raise ValueError("Qwen3 k_norm module is required")
            self.handles.append(attn.k_norm.register_forward_hook(self._hook(layer_id, "k")))
            self.handles.append(attn.v_proj.register_forward_hook(self._hook(layer_id, "v")))

    def _hook(self, layer_id, side):
        def capture(module, args, output):
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                raise RuntimeError("capture must be the real differentiable model output")
            output.retain_grad()
            self.values[layer_id, side] = output
        return capture

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def layer(self, layer_id, *, require_bf16=True):
        result = {}
        for side in ("k", "v"):
            value = self.values[layer_id, side]
            if value.grad is None:
                raise RuntimeError("captured KV output did not participate in CE backward")
            if require_bf16 and value.dtype != torch.bfloat16:
                raise ValueError("E0 model feature collection must be BF16")
            # A reshape changes layout only: no norm, RoPE, mean, W or scale here.
            feature = value.detach().reshape(value.shape[0], value.shape[1], -1).cpu()
            signed = value.grad.detach().float().reshape_as(feature).cpu()
            if not torch.isfinite(feature).all() or not torch.isfinite(signed).all():
                raise ValueError("nonfinite calibration feature or gradient")
            result[side + "_feature"] = feature.contiguous()
            result[side + "_grad_signed_fp32"] = signed.contiguous()
            result[side + "_grad_sq_fp32"] = signed.square().contiguous()
        return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def model_numeric_identity(model):
    """BF16 parameters/features must not demote the model's FP32 RoPE frequencies."""
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("E0 requires BF16 model parameters")
    frequency = model.model.rotary_emb.inv_freq
    if frequency.dtype != torch.float32:
        raise ValueError("E0 RoPE inverse frequencies must retain original FP32 precision; do not cast the whole module")
    raw = frequency.detach().cpu().contiguous().numpy().tobytes()
    return dict(model_parameter_dtype="bfloat16", rotary_inv_freq_dtype="float32",
                rotary_inv_freq_sha256=hashlib.sha256(raw).hexdigest())


def collect_windows(model, windows, output_dir, input_manifest, start=0, stop=64):
    """Each disjoint worker owns global window IDs; terminal JSON commits a window."""
    if not (0 <= start < stop <= len(windows)):
        raise ValueError("invalid global calibration window range")
    numeric_identity = model_numeric_identity(model)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    capture = Qwen3KVGraphCapture(model)
    model.eval()
    model.config.use_cache = False
    model_hash = _sha(Path(model.config._name_or_path)/"config.json")
    code_hash = _sha(__file__)
    try:
        for window_id in range(start, stop):
            dest = root / ("window_%03d" % window_id)
            # Resume only with full identity and every layer hash, not directory existence.
            terminal = dest/"complete.json"
            identity = dict(window_id=window_id, model_config_sha256=model_hash,
                            input_ids_sha256=input_manifest["input_ids_sha256"],
                            collector_sha256=code_hash, objective="full_window_shifted_ce_mean",
                            k_coordinate="post_k_norm_pre_rope", v_coordinate="actual_v",
                            model_dtype="bfloat16", signed_gradient_dtype="float32",
                            compensation="none_before_native_quantizer",
                            **numeric_identity)
            if terminal.exists():
                old = json.loads(terminal.read_text())
                if any(old.get(k) != v for k, v in identity.items()):
                    raise ValueError("existing Fisher window has a different identity")
                if len(old["layers"]) != len(model.model.layers):
                    raise ValueError("incomplete Fisher layer manifest")
                if not all(_sha(dest/r["file"]) == r["sha256"] for r in old["layers"]):
                    raise ValueError("existing Fisher layer hash mismatch")
                continue
            dest.mkdir(exist_ok=True)
            start_time = time.time()
            model.zero_grad(set_to_none=True)
            capture.clear()
            x = windows[window_id].to(model.get_input_embeddings().weight.device)
            outputs = model(input_ids=x, labels=x, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite calibration CE")
            loss.backward()
            records = []
            for layer_id in range(len(model.model.layers)):
                data = capture.layer(layer_id)
                # Record signed gradients explicitly, then their FP32 squares.
                data.update(window_id=window_id, layer_id=layer_id)
                filename = "layer_%02d.pt" % layer_id
                path = dest/filename
                temp = dest/(filename+".tmp")
                torch.save(data, temp)
                os.replace(temp, path)
                records.append(dict(file=filename, sha256=_sha(path), bytes=path.stat().st_size,
                                    k_shape=list(data["k_feature"].shape),
                                    v_shape=list(data["v_feature"].shape)))
                del data
            identity.update(loss=float(loss.detach()), input_tokens=x.shape[1],
                            effective_ce_tokens=x.shape[1]-1, layers=records,
                            duration_seconds=time.time()-start_time)
            tmp = dest/"complete.json.tmp"
            tmp.write_text(json.dumps(identity, indent=2)+"\n")
            os.replace(tmp, terminal)
            print(json.dumps({"window_id":window_id, "loss":identity["loss"],
                              "seconds":identity["duration_seconds"]}), flush=True)
            del outputs, loss, x
            capture.clear()
            model.zero_grad(set_to_none=True)
    finally:
        capture.close()


class FrozenFisherLayers:
    def __init__(self, root, *, expected_input_sha256):
        self.root = Path(root)
        self.manifests = []
        for window_id in range(64):
            row = json.loads((self.root/("window_%03d" % window_id)/"complete.json").read_text())
            if row["window_id"] != window_id or row["input_ids_sha256"] != expected_input_sha256:
                raise ValueError("Fisher input identity mismatch")
            if row["k_coordinate"] != "post_k_norm_pre_rope" or len(row["layers"]) != 36:
                raise ValueError("Fisher coordinate or layer count mismatch")
            self.manifests.append(row)
        self._cache_layer = None
        self._cache = {}

    def __getitem__(self, key):
        # Preserve the original quantizer's model.layers.N.self_attn.k_proj.weight API.
        parts = key.split(".")
        layer_id = int(parts[2])
        side = "k" if parts[-2] == "k_proj" else "v"
        if parts[-2] not in ("k_proj", "v_proj"):
            raise KeyError(key)
        if layer_id != self._cache_layer:
            self._cache.clear()
            rows = {"k": [], "v": []}
            for window_id, manifest in enumerate(self.manifests):
                record = manifest["layers"][layer_id]
                path = self.root/("window_%03d" % window_id)/record["file"]
                if _sha(path) != record["sha256"]:
                    raise ValueError("Fisher layer file hash mismatch")
                data = torch.load(path, map_location="cpu", weights_only=True)
                if data["window_id"] != window_id or data["layer_id"] != layer_id:
                    raise ValueError("Fisher row mapping mismatch")
                for which in ("k", "v"):
                    signed = data[which+"_grad_signed_fp32"]
                    squared = data[which+"_grad_sq_fp32"]
                    if signed.dtype != torch.float32 or not torch.equal(signed.square(), squared):
                        raise ValueError("Fisher must square the saved signed FP32 gradient")
                    rows[which].append(squared)
            self._cache = {which:torch.cat(value,dim=1) for which,value in rows.items()}
            self._cache_layer = layer_id
        return self._cache[side]
