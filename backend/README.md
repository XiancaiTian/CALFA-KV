# CALFA-KV Shared Runtime

This directory provides the encoding, cache, attention, and numerical dependencies shared by joint optimization and FAKV.

- `python/` contains the SGLang serving runtime and the CALFA encoding, magnitude, centering, and radial-scaling implementations.
- `kvquant/` fits the initial codebooks using fixed features.

The method entry points are in `joint_optimization/` and `fakv/`, and the end-to-end entry point is in `pipeline/`. The shared infrastructure serves both training and inference. See SOURCES.md, LICENSE, and NOTICE in the repository root for provenance and copyright information.
