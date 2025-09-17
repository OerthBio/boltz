#!/usr/bin/env python3
"""
Batch Boltz inference over a directory of YAMLs.

For each YAML file in --yaml_dir, run Boltz `predict` R times with seeds
seed_base .. seed_base + R - 1. Each run sets CUDA_VISIBLE_DEVICES from --device,
passes the selected parameters, and then renames the output directory:
  <out_dir>/boltz_results_<PREFIX>  ->  <out_dir>/boltz_results_<PREFIX>-<SEED>

Defaults match your previous usage.

Usage examples:
  python boltz_batch_run.py --yaml_dir ./BH_set1 --runs 10 --seed-base 10
  python boltz_batch_run.py --yaml_dir ./BH_set1 --runs 5 --device 1 --diffusion_samples 20 --recycling_steps 6
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional


def find_yaml_files(yaml_dir: Path) -> List[Path]:
    files = sorted(
        [p for p in yaml_dir.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}]
    )
    return files


def build_boltz_cmd(
    yaml_path: Path,
    seed: int,
    diffusion_samples: int,
    recycling_steps: int,
    step_scale: float,
    use_potentials: bool,
    affinity_mw_correction: bool,
    out_dir_for_boltz: Path,
) -> List[str]:
    cmd = [
        "boltz",
        "predict",
        str(yaml_path),
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
        "--step_scale",
        str(step_scale),
        "--seed",
        str(seed),
        "--out_dir",
        str(out_dir_for_boltz),
    ]
    if use_potentials:
        cmd.append("--use_potentials")
    if affinity_mw_correction:
        cmd.append("--affinity_mw_correction")
    return cmd


def run_one(
    yaml_path: Path,
    seed: int,
    device: str,
    diffusion_samples: int,
    recycling_steps: int,
    step_scale: float,
    use_potentials: bool,
    affinity_mw_correction: bool,
    out_dir: Path,
) -> int:
    """
    Run a single Boltz job for (yaml_path, seed). Returns process return code.
    After it finishes, rename:
      base_out = <out_dir>/boltz_results_<PREFIX>
      final_out = <out_dir>/boltz_results_<PREFIX>-<seed>
    """
    prefix = yaml_path.stem  # file name without suffix
    base_out = out_dir / f"boltz_results_{prefix}"
    final_out = out_dir / f"boltz_results_{prefix}-{seed}"
    base_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_boltz_cmd(
        yaml_path=yaml_path,
        seed=seed,
        diffusion_samples=diffusion_samples,
        recycling_steps=recycling_steps,
        step_scale=step_scale,
        use_potentials=use_potentials,
        affinity_mw_correction=affinity_mw_correction,
        out_dir_for_boltz=base_out,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    print(f"\n=== RUN ===")
    print(f"YAML: {yaml_path}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print("Cmd :", " ".join(cmd))
    sys.stdout.flush()

    rc = subprocess.call(cmd, env=env)

    # Rename results folder (if produced)
    if base_out.exists():
        try:
            if final_out.exists():
                # Avoid clobbering: remove existing final_out (e.g., from a previous attempt)
                shutil.rmtree(final_out)
            base_out.rename(final_out)
            print(f"Renamed: {base_out} -> {final_out}")
        except Exception as e:
            print(f"WARN: Could not rename {base_out} to {final_out}: {e}", file=sys.stderr)
    else:
        print(f"WARN: Expected output dir not found: {base_out}", file=sys.stderr)

    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch Boltz inference across YAML files with multi-seed runs."
    )
    ap.add_argument("--yaml_dir", required=True, help="Directory containing .yaml/.yml files.")
    ap.add_argument("--runs", type=int, default=10, help="Independent runs per YAML (default: 10).")
    ap.add_argument("--seed-base", type=int, default=10, help="Starting seed (default: 10).")
    ap.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value (default: 0).")
    ap.add_argument("--diffusion_samples", type=int, default=10, help="Boltz: diffusion_samples (default: 10).")
    ap.add_argument("--recycling_steps", type=int, default=10, help="Boltz: recycling_steps (default: 10).")
    ap.add_argument("--use_potentials", action="store_true", default=True, help="Boltz: use_potentials flag (default: True).")
    ap.add_argument("--no-use_potentials", dest="use_potentials", action="store_false", help="Disable use_potentials.")
    ap.add_argument("--step_scale", type=float, default=1.5, help="Boltz: step_scale (default: 1.5).")
    ap.add_argument("--affinity_mw_correction", action="store_true", default=False, help="Boltz: affinity_mw_correction flag (default: False).")
    ap.add_argument("--out_dir", default=".", help="Parent output directory (default: current dir).")
    args = ap.parse_args()

    yaml_dir = Path(args.yaml_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not yaml_dir.is_dir():
        print(f"ERROR: --yaml_dir not found: {yaml_dir}", file=sys.stderr)
        return 2

    files = find_yaml_files(yaml_dir)
    if not files:
        print(f"No YAML files found in {yaml_dir}.")
        return 0

    # Basic sanity check: boltz on PATH?
    if shutil.which("boltz") is None:
        print("ERROR: 'boltz' executable not found on PATH. Activate your env (e.g., 'conda activate boltz').", file=sys.stderr)
        return 2

    print(f"Found {len(files)} YAML files in {yaml_dir}")
    print(f"Runs per YAML: {args.runs}, seed_base: {args.seed_base}, device: {args.device}")
    print(f"Output parent dir: {out_dir}")
    print(f"Boltz params: diffusion_samples={args.diffusion_samples}, recycling_steps={args.recycling_steps}, "
          f"step_scale={args.step_scale}, use_potentials={args.use_potentials}, "
          f"affinity_mw_correction={args.affinity_mw_correction}")

    overall_rc = 0
    for f in files:
        for i in range(args.runs):
            seed = args.seed_base + i
            rc = run_one(
                yaml_path=f,
                seed=seed,
                device=args.device,
                diffusion_samples=args.diffusion_samples,
                recycling_steps=args.recycling_steps,
                step_scale=args.step_scale,
                use_potentials=args.use_potentials,
                affinity_mw_correction=args.affinity_mw_correction,
                out_dir=out_dir,
            )
            if rc != 0:
                print(f"WARN: boltz failed for {f.name} (seed {seed}) with code {rc}; continuing...", file=sys.stderr)
                overall_rc = rc  # remember a failure, but keep going

    print("Done.")
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())

