"""
scripts/run_experiments.py
============================
Master runner for TraceFlow experiments.

Usage
-----
    # dry-run preview of all experiments in smoke mode:
    python -m scripts.run_experiments --all --smoke --dry-run

    # run only exp01 in smoke mode:
    python -m scripts.run_experiments --only exp01 --smoke

    # run exp04 with custom attack settings:
    python -m scripts.run_experiments --only exp04 --smoke --attack latent --steps 3

    # run multiple experiments:
    python -m scripts.run_experiments --only exp01,exp04 --smoke

    # run all experiments (full, CUDA-scale):
    python -m scripts.run_experiments --all --output-dir results/traceflow

Flags
-----
    --all             Run all registered experiments.
    --only EXP[,EXP]  Run only the listed experiments (comma-separated IDs).
    --smoke           Enable smoke mode (fast, random data, DiT-XS).
    --dry-run         Print commands without executing them.
    --resume PATH     Resume training from the given checkpoint path.
    --output-dir DIR  Base directory for experiment outputs (default: results/traceflow).

Passthrough flags (forwarded to applicable experiments)
---------------------------------------------------
    --attack  {latent,pixel,both}     Inversion attack type   (exp04, exp05)
    --steps   INT                     Inversion attack steps  (exp04, exp05)
    --attacker {no_key,oracle_key,both}  Attacker knowledge   (exp04)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from experiments import REGISTRY

# Ordered list of all experiment IDs
ALL_EXPERIMENTS: List[str] = ["exp01", "exp02", "exp03", "exp04", "exp05"]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TraceFlow experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Selection
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--all", action="store_true",
        help="Run all registered experiments in order.",
    )
    sel.add_argument(
        "--only", type=str, default=None, metavar="EXP[,EXP]",
        help="Comma-separated experiment IDs to run (e.g. exp01,exp04).",
    )

    # Global flags
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: random data, DiT-XS, 3 training steps.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument(
        "--resume", default=None,
        help="Resume training from the given checkpoint path.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default="results/traceflow",
        help="Base directory for experiment outputs (default: results/traceflow).",
    )

    # Passthrough to inversion experiments (exp04, exp05)
    p.add_argument(
        "--attack", default="latent",
        choices=["latent", "pixel", "both"],
        help="Inversion attack type for exp04/exp05 (default: latent).",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Inversion attack steps for exp04/exp05 (default: 3 smoke, 50 full).",
    )
    p.add_argument(
        "--attacker", default="no_key",
        choices=["no_key", "oracle_key", "both"],
        help="Attacker knowledge level for exp04 (default: no_key).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_experiments(args: argparse.Namespace) -> List[str]:
    if args.all:
        return list(ALL_EXPERIMENTS)
    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [i for i in ids if i not in REGISTRY]
        if unknown:
            print(f"[runner] ERROR: unknown experiment IDs: {unknown}")
            print(f"[runner] Available: {', '.join(ALL_EXPERIMENTS)}")
            sys.exit(1)
        return ids
    # No selection flag → run all
    return list(ALL_EXPERIMENTS)


def _build_exp_args(
    global_args: argparse.Namespace,
    output_base: Path,
) -> argparse.Namespace:
    """Build the per-experiment args namespace from global runner args."""
    return argparse.Namespace(
        smoke=global_args.smoke,
        dry_run=global_args.dry_run,
        output_dir=str(output_base),
        # Passthrough
        attack=global_args.attack,
        steps=global_args.steps,
        attacker=global_args.attacker,
        resume=global_args.resume,
        # config=None lets each experiment use its own default
        config=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    to_run = _select_experiments(args)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    mode_str = "smoke" if args.smoke else "full"
    dry_str = " [DRY-RUN]" if args.dry_run else ""
    print(f"[runner] TraceFlow experiment runner{dry_str}")
    print(f"[runner] Mode:       {mode_str}")
    print(f"[runner] Output dir: {output_base}")
    print(f"[runner] Experiments: {', '.join(to_run)}")

    exp_args = _build_exp_args(args, output_base)

    summary: Dict[str, Any] = {}
    all_ok = True

    for exp_id in to_run:
        print(f"\n[runner] {'='*60}")
        print(f"[runner] Starting {exp_id}")
        print(f"[runner] {'='*60}")

        module_path = REGISTRY[exp_id]
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            print(f"[runner] ERROR: could not import {module_path}: {exc}")
            summary[exp_id] = {"status": "import_error", "error": str(exc)}
            all_ok = False
            continue

        try:
            result = mod.run(exp_args)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[runner] ERROR: {exp_id} raised: {exc}")
            summary[exp_id] = {"status": "exception", "error": str(exc)}
            all_ok = False
            continue

        summary[exp_id] = {
            "status": result.get("status", "unknown"),
            "output_dir": result.get("output_dir"),
        }
        if result.get("status") not in ("ok", "dry_run"):
            all_ok = False

    # Write summary
    summary_path = output_base / f"run_summary_{'smoke' if args.smoke else 'full'}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[runner] {'='*60}")
    print(f"[runner] Summary")
    print(f"[runner] {'='*60}")
    for exp_id, res in summary.items():
        status_str = res.get("status", "?")
        out = res.get("output_dir", "")
        print(f"  {exp_id}: {status_str}  {out}")

    print(f"\n[runner] Summary written: {summary_path}")
    print(f"[runner] All OK: {all_ok}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
