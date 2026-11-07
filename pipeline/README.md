# CALFA-KV End-to-End Workflow

The two core components are joint optimization and FAKV Anchor56. This directory connects their execution. Train and freeze a set of assets, use those same assets to calibrate Anchor Fisher, then launch attention with both quantized history and Anchors.

## One configuration for the full workflow

```bash
python3 pipeline/run.py --config pipeline/config.example.json \
  --start anchor --through serve --plan
```

`--plan` only prints the commands. It generates no assets and leaves CUDA uninitialized. Set the model and output paths in [config.example.json](config.example.json) to your local paths, then remove `--plan` to execute the workflow. The example starts from the bundled `assets/optimized/`. Use a new directory for the Anchor output.

| Scope | Arguments | Operations |
|---|---|---|
| Rerun joint calibration from initialization assets | `--start joint --through anchor` | Train and replay all 36 layers → export all layers → calibrate Anchors → verify assets |
| Calibrate from frozen assets | `--start anchor --through anchor` | Verify codebooks → calibrate Fisher → verify Fisher, codebook, and coordinate identities |
| Run the full workflow through serving | `--start joint --through serve` | Joint optimization, export, Anchor calibration, verification, and server startup |
| Serve with existing matching Anchor assets | `--start serve --through serve` | Verify the assets and start the server |

For retraining, add the following fields to the configuration.

```json
{
  "joint": {
    "initial_assets": "../joint_optimization/coordinates/initialization",
    "runs": "/path/to/new_runs",
    "configs": ["/path/to/layer_00.json", "/path/to/layer_01.json"]
  }
}
```

The example shows the field structure. The actual `configs` list must contain 36 distinct configurations in layer order. See the [joint optimization configuration schema](../joint_optimization/config.schema.json) for per-layer fields. Each layer's model, initialization directory, output directory, and token inputs must agree with the top-level configuration. A new training run requires per-window captures. Generate the large model activation captures separately.

Training and replay run in separate processes. Export requires 36 valid per-layer verification records. Use `--through joint` to stop after joint training and export. If a subcommand fails, the workflow stops before subsequent calibration or serving. Use `--start` to select the stage after existing outputs. Only stages actually run count as executed.

## Shared runtime

[backend/](../backend/) serves both native output verification for joint optimization and FAKV inference. Training runs with Anchors disabled. The serving entry point configures CALFA-KV directly.

| Code | Integration point |
|---|---|
| [qwen3.py](../backend/python/sglang/srt/models/qwen3.py) | Binds the actual RoPE implementation and submits prefill quantization candidates |
| [unified_kv_pool.py](../backend/python/sglang/srt/mem_cache/unified_kv_pool.py) | Shared encoding assets, original Recent K, and the Anchor lifecycle |
| [common.py](../backend/python/sglang/srt/mem_cache/common.py) | Processes migration candidates after page-8 encoding |
| [quantized_kv_prefill.py](../backend/python/sglang/srt/layers/attention/quantized_kv_prefill.py) | Restores the center only for quantized rows |
| [triton_backend.py](../backend/python/sglang/srt/layers/attention/triton_backend.py) | Anchor replacement during prefill and decode metadata |
| [decode_attention.py](../backend/python/sglang/srt/layers/attention/triton_ops/decode_attention.py) | Uses Anchors for matching rows and bypasses the quantization scale and center terms |

See [pyproject.toml](../backend/python/pyproject.toml) for dependencies. Focused kernel verification used Triton 3.5.1. Full serving also requires compatible native SGLang dependencies. The [test source](../tests/test_calfa.py) and [VALIDATION.json](../VALIDATION.json) list stubs for unrelated imports, numerical tolerances, and test scope. Full serving and v9 accuracy verification for the combined configuration are pending.
