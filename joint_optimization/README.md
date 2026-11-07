# CALFA-KV Joint Optimization

Joint optimization is the first part of the method. It combines channel coordinates, dynamic magnitude, grouping, and codebook construction in one offline recipe, then uses post-projection attention output error to jointly learn the center, radial parameters, and K/V codebooks. FAKV Anchor56 uses the resulting frozen assets.

## Offline recipe

| Operation | Purpose | Code |
|---|---|---|
| Coordinate and quantization initialization | Derive D from query energy and define the encoding features with the stored FP16 N. Use covariance to match RoPE pairs and fit finite-capacity codebooks | [coordinates/](coordinates/), [prepare_coordinates.py](tools/prepare_coordinates.py), [backend/kvquant/](../backend/kvquant/) |
| Joint grouping and codebook selection | Run two rounds of finite-rate regrouping. Fit K codebooks for swaps and matched no-swap controls with the same budget, then retain the selected P/C | [finite_rate_groups.py](coordinates/finite_rate_groups.py), [grouping.py](core/grouping.py), [grouping_rules.py](core/grouping_rules.py) |
| K/V codebook warm start | Fix D/P and the magnitude rule, then optimize K/V under a shared output error objective | [initialize_codebooks.py](core/initialize_codebooks.py), [kv_objective.py](core/kv_objective.py) |
| Continuous joint calibration | Independently update the center, pair-shared radial parameters, and K/V codebooks in each layer, then reselect codes and evaluate candidates using native outputs | [problem.py](core/problem.py), [objective.py](core/objective.py), [calibration.py](core/calibration.py) |
| Solve, accept, and freeze | Use damped GN/CG, halving backtracking, and FIT/validation criteria, then replay each layer and export all layers | [direction.py](core/direction.py), [gauss_newton.py](core/gauss_newton.py), [policy.py](core/policy.py), [export.py](export.py) |

These operations belong to one joint optimization stage and follow their numerical dependencies, independent of historical experiment versions. Discrete grouping first selects P/C. The final continuous calibration then fixes D/P and updates μ/A/Cᴷ/Cⱽ together. Each operation runs at its own point in the recipe.

The initial assets are in [coordinates/initialization/](coordinates/initialization/), and the frozen outputs are in [assets/optimized/](../assets/optimized/). The initial assets use a zero center and radial weights initialized to D. The frozen outputs contain the learned center and radial parameters. Both sets use the current CALFA field format. The migration changed only fields and identifiers. See [FORMAT_MIGRATION.json](../assets/FORMAT_MIGRATION.json) for verification that the numerical values were preserved.

## Training and replay entry point

```bash
python3 joint_optimization/optimize.py --config /path/to/layer_config.json --validate-only
python3 joint_optimization/optimize.py --config /path/to/layer_config.json
```

The default `--stage all` runs training and exact replay in sequence in two separate processes. Use `--stage train` or `--stage verify` to run either stage individually. Joint calibration runs with Anchors disabled. Both components share the encoding and attention kernels in [backend/python/](../backend/python/).

See [config.schema.json](config.schema.json) for the configuration fields. Relative paths resolve against the directory containing the configuration file.

| Field | Requirement |
|---|---|
| `model`, `assets` | Fixed local model and a zero-center initialization directory containing `bundle/` |
| `layer`, `gpu`, `port` | Layer index 0–35, GPU index in the current visible-device list, and distributed initialization port |
| `input_ids`, `input_ids_sha256` | 64 fixed 2056-token windows and the file hash |
| `teacher_layers` | 64 captures for the current layer containing K/V, queries, and reference outputs from the same frozen model state, each with a file and hash |
| `cache_metadata` | Actual quantized history lengths for the same layer, windows, and query positions, each with a file and hash |
| `projection` | Safetensors file, key, and file hash for the current layer's output projection |
| `output` | New directory, typically `runs/layer_00` through `runs/layer_35` |

Reference captures include `k_pre_rope_full`, `v_live_full`, `q_post_rope`, and `attention_output_post_wo`. See [problem.py](core/problem.py) for checks on tensor shapes, activation precision, positions, and actual cache lengths. Coordinate and grouping initialization use the original 2048-token inputs. Final continuous calibration uses the template-wrapped 2056-token inputs. The input files are included. Generate the large captures separately.

## Export

```bash
python3 joint_optimization/export.py \
  --initial-assets joint_optimization/coordinates/initialization \
  --runs /path/to/runs \
  --output /path/to/new_calfa_bundle
```

Export requires completed training and independent GPU replay for each layer. It checks the selected files, layer indices, D/P, and parameter formats, then merges only the selected results. The output feeds directly into the Anchor Fisher calibration in Part 2. [pipeline/run.py](../pipeline/run.py) can run the full workflow automatically.

Model weights are fixed throughout this stage, and each of the 36 layers is calibrated independently. This consolidation did not rerun joint training or v9 evaluation.
