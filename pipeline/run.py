#!/usr/bin/env python3
"""Run CALFA-KV joint optimization, Anchor calibration and serving as one pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("joint", "anchor", "serve")


def commands(config_path, start, through):
    """Build commands without creating artifacts or starting CUDA."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("method") != "CALFA-KV":
        raise ValueError("Expected a CALFA-KV pipeline configuration")
    if STAGES.index(start) > STAGES.index(through):
        raise ValueError("Start must precede the requested final stage")

    def path(value):
        return (config_path.parent / value).resolve()

    model, bundle = path(config["model"]), path(config["bundle"])
    anchor = config["anchor"]
    anchor_dir = path(anchor["output"])
    manifest = anchor_dir / "manifest.json"
    result = []

    def add(stage, *args):
        result.append({"stage": stage, "argv": [str(arg) for arg in args]})

    if start == "joint":
        joint = config["joint"]
        configs = [path(item) for item in joint["configs"]]
        runs, initial = path(joint["runs"]), path(joint["initial_assets"])
        if len(configs) != 36 or len(set(configs)) != 36:
            raise ValueError("Joint optimization requires 36 distinct layer configurations")
        for layer, file in enumerate(configs):
            row = json.loads(file.read_text())
            if row["layer"] != layer:
                raise ValueError("Layer configurations must be ordered from 0 to 35")
            expected = {"model": model, "assets": initial,
                        "output": runs / ("layer_%02d" % layer),
                        "input_ids": path(anchor["input_ids"])}
            for name, value in expected.items():
                if (file.parent / row[name]).resolve() != value:
                    raise ValueError("Layer configuration disagrees with pipeline: " + name)
            add("joint", sys.executable, ROOT / "joint_optimization/optimize.py",
                "--config", file, "--stage", "all")
        add("joint", sys.executable, ROOT / "joint_optimization/export.py",
            "--initial-assets", initial, "--runs", runs, "--output", bundle)
    if STAGES.index(start) <= STAGES.index("anchor") <= STAGES.index(through):
        add("anchor", sys.executable, ROOT / "pipeline/verify_assets.py", "--bundle", bundle)
        add("anchor", sys.executable, ROOT / "fakv/calibrate_fisher.py",
            "--model", model, "--bundle", bundle,
            "--input-ids", path(anchor["input_ids"]), "--output", anchor_dir)
        add("anchor", sys.executable, ROOT / "pipeline/verify_assets.py",
            "--bundle", bundle, "--anchor-manifest", manifest)
    if through == "serve":
        add("serve", "bash", ROOT / "pipeline/serve.sh", model, bundle, manifest,
            *config.get("serve_args", []))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--start", choices=STAGES, required=True)
    parser.add_argument("--through", choices=STAGES, default="anchor")
    parser.add_argument("--plan", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    plan = commands(args.config, args.start, args.through)
    if args.plan:
        print(json.dumps({"method": "CALFA-KV", "commands": plan, "executed": False}, indent=2))
        return
    env = os.environ.copy()
    env["SGLANG_CALFA_ANCHOR_ENABLE"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for command in plan:
        print("CALFA-KV:", command["stage"], command["argv"], flush=True)
        subprocess.run(command["argv"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
