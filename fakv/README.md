# CALFA-KV FAKV Anchor56

FAKV uses the frozen assets produced by joint optimization, calibrates Anchor Fisher separately, and maintains up to 56 high-precision positions in quantized history per request, layer, and KV head.

| File | Purpose |
|---|---|
| [calibrate_fisher.py](calibrate_fisher.py) | Computes Fisher weights in the final coordinates from K gradients after k_norm and before RoPE, and from V gradients. Binds the calibration to the final asset hashes |
| [calfa_anchor_coordinates.py](calfa_anchor_coordinates.py) | Signed gradient transforms, Fisher regularization, coordinate identity, the error reference, and center restoration for quantized prefixes |
| [calfa_anchor_codec.py](calfa_anchor_codec.py) | Reconstruction from native packed G8/256 indices, nonzero center restoration, and error scoring at the actual positions |
| [calfa_anchor_cache.py](calfa_anchor_cache.py) | Bounded Top56 directory, stable tie-breaking, paired FP16 K/V, and request cleanup and isolation |
| [calfa_anchor_runtime.py](calfa_anchor_runtime.py) | Asset verification, prefill and migration candidates, prefix replacement, and decode parameters |

The encoding coordinates use fixed D/P and the stored N, which depends on the final center and radial parameters. The center cancels in the reconstruction error, and transformed signed gradients are squared before aggregation. Anchor Fisher is separate from the Fisher weights used to fit the initial codebooks.

An Anchor hit uses physical pre-RoPE FP16 K/V. K is rotated at its actual position and bypasses the quantization scale and center restoration. V directly replaces its reconstructed row. The request, position, and live slot must all match. Candidate reconstruction and native decode use the same explicit FMA order for center rotation to keep FP16 rounding consistent.

See the [main README](../README.md) for the full equations, calibration instructions, and serving commands. Anchor Fisher and v9 results for the combined configuration are pending. Calibrate Fisher for the final assets before evaluating the combination.
