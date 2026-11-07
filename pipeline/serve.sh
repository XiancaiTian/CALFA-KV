#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -lt 3 ]]; then
  echo 'Usage: bash pipeline/serve.sh MODEL_PATH BUNDLE_DIR ANCHOR_MANIFEST [SGLang arguments...]'
  echo 'CALFA-KV: frozen joint assets, G8/256, Recent64, paired FP16 Anchor56.'
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
calfa_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
calfa_model=$1
calfa_bundle=$(realpath -- "$2")
calfa_manifest=$(realpath -- "$3")
shift 3
python3 "$calfa_root/pipeline/verify_assets.py" --bundle "$calfa_bundle" --anchor-manifest "$calfa_manifest"
export PYTHONPATH="$calfa_root/fakv:$calfa_root/backend/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_VQ_PRE_ROPE=1
export SGLANG_VQ_CODEBOOK_PATH="$calfa_bundle/codebook.pt"
export SGLANG_VQ_V_CODEBOOK_PATH="$calfa_bundle/codebook_v.pt"
export SGLANG_CALFA_K_ROTATION_PATH="$calfa_bundle/k_rotation_qqt_r_h_pbr.pt"
export SGLANG_CALFA_V_ROTATION_PATH="$calfa_bundle/v_rotation_sst_r_h_pbr.pt"
export SGLANG_CALFA_ABSORB_V_ROTATION=1
export SGLANG_MIXED_KV_PREFIX_TOKENS=0 SGLANG_MIXED_KV_RECENT_TOKENS=64
export SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float16
export SGLANG_V_INT_BITS=2 SGLANG_V_INT_MAX_Q=3
export SGLANG_VQ_FP8_FMT=e5m2
export SGLANG_VQ_OPT_QMAP=1 SGLANG_VQ_OPT_FLUSH=1
export SGLANG_VQ_OPT_PREFILL=1 SGLANG_VQ_OPT_KMAP=1
export SGL_VQWIDE_TRIG_ONFLY=1
export SGLANG_VQ2_CUDA=0 SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL=0
export SGLANG_CALFA_ANCHOR_ENABLE=1
export SGLANG_CALFA_ANCHOR_ASSET_MANIFEST="$calfa_manifest"
SGLANG_CALFA_ANCHOR_FISHER=$(python3 - "$calfa_manifest" <<'FISHER_PATH'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads(p.read_text())
print((p.parent/m['assets']['anchor_fisher']['path']).resolve())
FISHER_PATH
)
export SGLANG_CALFA_ANCHOR_FISHER
exec python3 -m sglang.launch_server \
  --model-path "$calfa_model" --trust-remote-code --host 127.0.0.1 \
  --prefill-attention-backend triton --decode-attention-backend triton \
  --sampling-backend pytorch --tp-size 1 --max-running-requests 16 \
  --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
  --triton-attention-num-kv-splits 48 --disable-radix-cache \
  --dtype bfloat16 --context-length 32768 --chunked-prefill-size -1 \
  --disable-piecewise-cuda-graph --mem-fraction-static 0.8 "$@"
