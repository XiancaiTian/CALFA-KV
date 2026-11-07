"""Read frozen native gpqa_code64 inputs for CALFA calibration."""
import hashlib
import json
import os
from pathlib import Path
import torch


def frozen_windows(directory=None, *, nsamples=64, seqlen=2048, seed=0, code_n=16, require_admitted=True):
    directory = directory or os.environ.get("CALFA_FROZEN_CALIB_DIR")
    if not directory:
        raise RuntimeError("CALFA_FROZEN_CALIB_DIR is required: no live resampling")
    if (nsamples, seqlen, seed, code_n) != (64, 2048, 0, 16):
        raise ValueError("CALFA input contract is 48 GPQA +16 code windows, 2048 tokens, seed0")
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if require_admitted and not manifest.get("formal_admitted"):
        raise RuntimeError("Frozen inputs have not passed the pinned-environment admission")
    raw = (root / "input_ids.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["input_ids_sha256"]:
        raise ValueError("frozen calibration input hash mismatch")
    rows = json.loads(raw)
    if len(rows) != 64 or [r["window_id"] for r in rows] != list(range(64)):
        raise ValueError("invalid global calibration window ordering")
    if [r["source"] for r in rows] != ["gpqa"]*48 + ["code"]*16:
        raise ValueError("invalid native corpus mixture")
    windows = [torch.tensor(r["input_ids"], dtype=torch.int64).view(1, -1) for r in rows]
    if any(tuple(x.shape) != (1, 2048) for x in windows):
        raise ValueError("invalid frozen window length")
    return windows, manifest


def frozen_loader(**kwargs):
    windows, _ = frozen_windows(**kwargs)
    rows = []
    for inp in windows:
        target = inp.clone()
        target[:, :-1] = -100  # Original loader convention; Fisher uses labels=inp.
        rows.append((inp, target))
    return rows, None
