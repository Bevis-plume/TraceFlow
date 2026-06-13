"""
experiments/common.py
======================
Shared utilities for the TraceFlow experiment framework.

Provides:
  - run_command()      — safe subprocess execution with dry-run support
  - load_json()        — JSON loader that returns {} on missing/malformed
  - write_json()       — atomic-ish JSON writer
  - read_last_jsonl()  — last line of a JSONL log file
  - append_csv_row()   — CSV appender with auto-header
  - make_exp_dir()     — per-experiment run directory creator
  - copy_config()      — copy YAML config into run dir as run_config.yml
  - train_output_dir() — canonical training artifact directory for a run_name
  - checkpoint_path()  — canonical checkpoint path for a run_name
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------

def python_module_cmd(module: str) -> List[str]:
    """Return an unbuffered Python module command for experiment subprocesses."""
    return [sys.executable, "-u", "-B", "-m", module]


def _unbuffered_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _normalise_python_cmd(cmd: List[str]) -> List[str]:
    """Upgrade ``python -m module`` commands to unbuffered ``python -u -B -m``."""
    if not cmd or Path(cmd[0]).name != Path(sys.executable).name:
        return cmd
    if "-m" not in cmd:
        return cmd
    module_idx = cmd.index("-m")
    module = cmd[module_idx + 1] if module_idx + 1 < len(cmd) else None
    if module is None:
        return cmd
    return [cmd[0], "-u", "-B", "-m", module] + cmd[module_idx + 2:]

def run_command(
    cmd: List[str],
    *,
    dry_run: bool = False,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run *cmd* as a subprocess with live stdout/stderr.

    Returns a dict with keys ``returncode`` and ``dry_run``.
    On ``dry_run=True`` the command is printed but not executed (returns 0).
    """
    cmd = _normalise_python_cmd(cmd)
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  $ {cmd_str}")
    if dry_run:
        print("  [dry-run] skipped")
        return {"returncode": 0, "dry_run": True}
    result = subprocess.run(cmd, cwd=cwd, env=_unbuffered_env())
    return {"returncode": result.returncode, "dry_run": False}


# ---------------------------------------------------------------------------
# JSON / JSONL helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file.  Returns an empty dict if the file is missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_last_jsonl(path: Path) -> Dict[str, Any]:
    """Return the last non-empty line of a JSONL file as a dict."""
    try:
        last: Optional[str] = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last:
            return json.loads(last)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append *row* to a CSV file, writing the header if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(row.keys()), extrasaction="ignore"
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)



def _redact_secrets_for_copy(obj: Any) -> Any:
    """Redact private key material before copying configs into public results."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if str(key).lower() in {"secret", "secret_key", "private_key"}:
                out[key] = "REDACTED"
            else:
                out[key] = _redact_secrets_for_copy(value)
        return out
    if isinstance(obj, list):
        return [_redact_secrets_for_copy(v) for v in obj]
    return obj

# ---------------------------------------------------------------------------
# Directory / config helpers
# ---------------------------------------------------------------------------

def make_exp_dir(output_base: Path, exp_id: str, smoke: bool) -> Path:
    """Create and return ``<output_base>/<exp_id>/<smoke|full>/``."""
    run_dir = output_base / exp_id / ("smoke" if smoke else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_config(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* as ``run_config.yml`` with secrets redacted."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(src) as f:
            cfg = yaml.safe_load(f) or {}
        with open(dest_dir / "run_config.yml", "w") as f:
            yaml.safe_dump(_redact_secrets_for_copy(cfg), f, sort_keys=False, allow_unicode=True)
    except Exception:
        shutil.copy2(src, dest_dir / "run_config.yml")


def _artifact_roots_from_config(config_path: Optional[str]) -> Tuple[Path, Path]:
    """Return ``(output_root, checkpoint_root)`` from a resolved train config."""
    output_root = Path("outputs/flow_transformer")
    checkpoint_root = Path("checkpoints/flow_transformer")
    if not config_path:
        return output_root, checkpoint_root
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        output_root = Path(cfg.get("project", {}).get("output_dir", output_root))
        checkpoint_root = Path(cfg.get("training", {}).get("checkpoint_dir", checkpoint_root))
    except Exception:
        pass
    return output_root, checkpoint_root


def train_output_dir(
    run_name: str,
    base: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Canonical training output directory for *run_name*.

    Prefer the resolved config path when available so unified bundle runs keep
    all artifacts inside the requested downloadable directory.
    """
    output_root, _ = _artifact_roots_from_config(config_path)
    return Path(base) / run_name if base else output_root / run_name


def checkpoint_path(
    run_name: str,
    base: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Canonical latest-checkpoint path for *run_name*."""
    _, checkpoint_root = _artifact_roots_from_config(config_path)
    root = Path(base) if base else checkpoint_root
    return root / run_name / "latest.pt"


def resolve_checkpoint(
    *,
    run_name: str,
    config_path: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Path:
    """Resolve the checkpoint an experiment should use.

    ``explicit`` is used by eval-only workflows that reuse an already-trained
    checkpoint.  When omitted, the canonical checkpoint path for ``run_name``
    is returned so existing train-then-eval workflows keep their old behavior.
    """
    if explicit:
        return Path(explicit).expanduser()
    return checkpoint_path(run_name, config_path=config_path)


def infer_run_dir_from_checkpoint(
    checkpoint: Path,
    *,
    run_name: str,
    config_path: Optional[str] = None,
) -> Path:
    """Infer a training output directory from a checkpoint path.

    Bundle checkpoints follow ``<bundle>/checkpoints/<run_name>/latest.pt`` and
    their logs live at ``<bundle>/outputs/<run_name>``.  For arbitrary external
    checkpoints, fall back to the canonical output path from the active config.
    """
    ckpt = Path(checkpoint).expanduser()
    if ckpt.name.endswith(".pt") and ckpt.parent.name == run_name:
        ckpt_root = ckpt.parent.parent
        bundle_root = ckpt_root.parent
        if ckpt_root.name == "checkpoints":
            candidate = bundle_root / "outputs" / run_name
            if candidate.exists():
                return candidate
    return train_output_dir(run_name, config_path=config_path)


def checkpoint_exists(checkpoint: Path, *, dry_run: bool = False) -> bool:
    """Treat dry-runs as valid while checking real checkpoint existence otherwise."""
    return dry_run or Path(checkpoint).expanduser().exists()


def should_train_for_policy(train_policy: str, checkpoint: Path, *, dry_run: bool = False) -> bool:
    """Return whether an experiment should run training before evaluation."""
    policy = (train_policy or "always").lower()
    if policy == "always":
        return True
    if policy == "missing":
        return not checkpoint_exists(checkpoint, dry_run=dry_run)
    if policy == "never":
        return False
    raise ValueError(f"Unknown train_policy={train_policy!r}; expected always, missing, or never")


def write_skipped_metrics(
    *,
    exp_id: str,
    exp_dir: Path,
    output_base: Path,
    config_path: str,
    run_name: str,
    smoke: bool,
    checkpoint: Path,
    reason: str,
    validation_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write explicit skipped metrics for eval-only runs missing an optional ckpt."""
    metrics: Dict[str, Any] = {
        "exp_id": exp_id,
        "config": config_path,
        "smoke": smoke,
        "run_name": run_name,
        "status": "skipped_missing_checkpoint",
        "checkpoint": str(checkpoint),
        "reason": reason,
        "validation_warnings": validation_warnings or [],
    }
    write_json(exp_dir / "metrics.json", metrics)
    append_csv_row(output_base / "all_metrics.csv", metrics)
    print(f"[{exp_id}] skipped: {reason}")
    print(f"[{exp_id}] Metrics written: {exp_dir / 'metrics.json'}")
    return {"status": metrics["status"], "exp_id": exp_id, "metrics": metrics, "output_dir": str(exp_dir)}


def select_default_config(
    *,
    smoke: bool,
    override: Optional[str],
    smoke_config: str,
    full_config: str,
) -> str:
    """Resolve the experiment config path.

    Priority:
      1. Explicit ``override`` from the CLI / runner.
      2. ``smoke_config`` when ``smoke=True``.
      3. ``full_config`` otherwise.
    """
    if override:
        return override
    return smoke_config if smoke else full_config


def validate_experiment_config(
    *,
    config_path: str,
    run_name: str,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """Dry-run validation for reproducible experiment launches.

    Validation is advisory by default: it reports issues but does not abort the
    experiment runner, so smoke/full dry-runs still work on developer machines.
    """
    path = Path(config_path)
    results: Dict[str, Any] = {
        "config": str(path),
        "ok": True,
        "warnings": [],
        "checked": {
            "dataset_path": None,
            "secret_key": None,
            "diffusers_dependency": None,
            "output_dir": None,
            "checkpoint_dir": None,
            "resume": None,
        },
    }

    if not path.exists():
        results["ok"] = False
        results["warnings"].append(f"config file not found: {path}")
        return results

    with open(path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name")
    dataset_root = data_cfg.get("root")
    if dataset_name in {"imagefolder", "flat"} and dataset_root:
        dataset_path = Path(dataset_root)
        results["checked"]["dataset_path"] = str(dataset_path)
        if not dataset_path.exists():
            results["warnings"].append(
                f"dataset path missing for {dataset_name}: {dataset_path}"
            )
    else:
        results["checked"]["dataset_path"] = dataset_root

    lt_cfg = cfg.get("security", {}).get("latent_transform", {})
    secret_key = lt_cfg.get("secret_key")
    results["checked"]["secret_key"] = "set" if secret_key else "unset"
    if lt_cfg.get("type") == "keyed":
        if not secret_key:
            results["warnings"].append("keyed latent transform has no secret_key configured")
        elif any(token in str(secret_key) for token in ("CHANGE_ME", "DEV_ONLY", "FULL_TRACEFLOW_KEY_CHANGE_ME")):
            results["warnings"].append("secret_key still uses a placeholder value")

    ae_cfg = cfg.get("autoencoder", {})
    results["checked"]["diffusers_dependency"] = ae_cfg.get("backend")
    if ae_cfg.get("backend") == "diffusers":
        if importlib.util.find_spec("diffusers") is None:
            results["warnings"].append(
                "diffusers backend configured but 'diffusers' is not installed"
            )

    output_dir = Path(cfg.get("project", {}).get("output_dir", "outputs/flow_transformer")) / run_name
    checkpoint_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints/flow_transformer")) / run_name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    results["checked"]["output_dir"] = str(output_dir)
    results["checked"]["checkpoint_dir"] = str(checkpoint_dir)

    if resume:
        resume_path = Path(resume)
        results["checked"]["resume"] = str(resume_path)
        if not resume_path.exists():
            results["warnings"].append(f"resume checkpoint not found: {resume_path}")

    return results


def print_validation_report(exp_id: str, validation: Dict[str, Any]) -> None:
    """Print a concise dry-run validation report."""
    print(f"[{exp_id}] Validation: config={validation['config']}")
    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"[{exp_id}]   warning: {warning}")
    else:
        print(f"[{exp_id}]   OK: dataset path, key placeholder, dependency, and dirs checked")


# ---------------------------------------------------------------------------
# Seed (thin wrapper kept here so experiments don't import from src directly)
# ---------------------------------------------------------------------------

def seed_run(seed: int) -> None:
    """Set Python / NumPy / PyTorch global seeds."""
    import random
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
