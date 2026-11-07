# CALFA-KV Source Provenance and Compatibility

This document records code provenance and licensing. The main README presents CALFA-KV as a unified mathematical workflow. The method's new name leaves the original code's copyright, provenance, and scope of experimental evidence intact.

| Component | Source | How it is incorporated |
|---|---|---|
| Base grouped vector quantization, W/N/P, SGLang, and KVQuant | Official TaSQ commit `28532bc03266a87c4e079e2c3da52dab257c115a` and subsequent fixes in this repository | The implementation and licenses are retained under consistent public entry-point names. The original authors are credited for the underlying design |
| Grouping and initial codebook optimization, and joint center, radial, and K/V calibration across all 36 layers | Sibling directory `tasq-teacher-affine-full36`. Training source `68c4e84c50053561cec390d159b88c379355cea7` and serving source `2a2f5529672b0f6ffc6bd5e10151403499aaf3f8` | Incorporated into `joint_optimization/` and the supporting runtime, preserving the numerical workflow and frozen assets |
| Anchor56 cache and lifecycle | Sibling directory `tasq-fakv-anchor56` | Incorporated into `fakv/` and adapted for G8/256, a separate radial scale, a nonzero center, and Fisher weights in the final coordinates |
| Integration of the two components | This CALFA-KV integration | Joint asset and Anchor identity checks, actual reconstruction, center restoration for quantized prefixes, skipping redundant scale and center operations during decode, calibration and serving entry points, matching the FMA order for center rotation, and training, verification, and export entry points independent of historical jobs |

[UPSTREAM_SOURCE_MANIFEST.json](UPSTREAM_SOURCE_MANIFEST.json) preserves the original per-file manifest for training and runtime. It records the **upstream snapshots before consolidation**, using their original paths. [MANIFEST.json](MANIFEST.json) records the current layout and file identities.

## Current layout and asset formats

The package includes one shared `backend/`. The joint optimization algorithms used by CALFA-KV are in `joint_optimization/core/` and `coordinates/`, and FAKV is in `fakv/`. Historical experiment jobs, unused optimization variants, and launch entry points for other baselines are excluded. The old serialization fields and runtime prefixes have migrated to CALFA coordinate fields and naming. Legacy method aliases are unavailable.

Only fields and string identifiers changed when migrating the initialization and frozen assets. The shapes, precisions, and values of all 18,706 tensors and arrays across 9 files were preserved. See [assets/FORMAT_MIGRATION.json](assets/FORMAT_MIGRATION.json). Historical paths are kept for attribution. The grouping provenance of the initialization assets is stored in [SOURCE_ASSET_PROVENANCE.json](SOURCE_ASSET_PROVENANCE.json), outside the runtime asset fields.

Joint optimization uses the shared runtime with Anchors disabled. The source consolidation provides no evidence of new training, calibration, or v9 accuracy results.

## Copyright and citations

The LICENSE and NOTICE files and the original copyright headers are retained at the repository root and in the relevant source directories. The licenses for SGLang, KVQuant, and other third-party components still apply. Academic publications should cite the base quantization, the calibration and optimization methods used, and the Anchor-related sources separately to credit their respective contributions.
