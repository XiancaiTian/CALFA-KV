# CALFA-KV Experiment Entry Point

CALFA-KV has two core components, [joint optimization](joint_optimization/README.md) and [FAKV Anchor56](fakv/README.md). The [end-to-end entry point](pipeline/README.md) connects them.

The configuration is Qwen3-4B-Instruct-2507, G8/256, RoPE5M, and Prefix0/Recent64/Anchor56. See [README.md](README.md) for the method, workflow diagram, commands, and verification scope. Frozen joint optimization assets are included. The combined configuration still requires new runs to verify Fisher calibration, full model serving, and v9 accuracy.
