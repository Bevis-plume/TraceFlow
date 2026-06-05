"""
scripts.traceflow
=================
Unified server-friendly command-line entry for TraceFlow.

Public workflow entries:
  python -m scripts.traceflow train-generator --config configs/traceflow.yml
  python -m scripts.traceflow train-final     --config configs/traceflow.yml
  python -m scripts.traceflow run-all         --config configs/traceflow.yml
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from experiments import REGISTRY
from src.utils.config_composer import (
    apply_overrides,
    compose_run_config,
    compose_suite_experiment,
    deep_merge,
    load_yaml,
    redact_secrets,
    validate_resolved_config,
    write_yaml,
)

EXP_ORDER = ["exp01", "exp02", "exp03", "exp04", "exp05"]
BASE_SECTIONS = ["project", "data", "autoencoder", "model", "training", "sampling", "smoke", "security", "watermark"]
DEFAULT_CONFIG = "configs/traceflow.yml"


def _python_module_cmd(module: str) -> List[str]:
    """Return a Python module command with unbuffered stdout/stderr."""
    return [sys.executable, "-u", "-B", "-m", module]


def _unbuffered_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: List[str], *, dry_run: bool = False, log_path: Optional[Path] = None) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        print("  [dry-run] skipped", flush=True)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("$ " + " ".join(str(c) for c in cmd) + "\n[dry-run] skipped\n")
        return 0
    env = _unbuffered_env()
    if log_path is None:
        return subprocess.run(cmd, env=env).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
        return proc.wait()


def _resolved_dir(name: str) -> Path:
    safe = name.replace("/", "_").replace(" ", "_")
    return Path("local_configs") / "resolved" / safe


def _write_resolved_config(resolved: Dict[str, Any], dest: Path) -> Path:
    write_yaml(dest, resolved)
    write_yaml(dest.with_suffix(".redacted.yml"), redact_secrets(resolved))
    return dest


def _base_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: deepcopy(raw[k]) for k in BASE_SECTIONS if k in raw}


def _set_project(config: Dict[str, Any], *, name: str | None = None, output_root: str | None = None, checkpoint_root: str | None = None) -> None:
    if name:
        config.setdefault("project", {})["name"] = name
    if output_root:
        config.setdefault("project", {})["output_dir"] = output_root
    if checkpoint_root:
        config.setdefault("training", {})["checkpoint_dir"] = checkpoint_root


def _keyed_security(template: Mapping[str, Any]) -> Dict[str, Any]:
    lt = deepcopy(template.get("security", {}).get("latent_transform", {}))
    lt["type"] = "keyed"
    return {"latent_transform": lt}


def _identity_security(template: Mapping[str, Any]) -> Dict[str, Any]:
    lt = deepcopy(template.get("security", {}).get("latent_transform", {}))
    lt["type"] = "identity"
    lt["secret_key"] = "UNUSED_FOR_IDENTITY"
    return {"latent_transform": lt}


def _enabled_watermark(template: Mapping[str, Any]) -> Dict[str, Any]:
    wm = deepcopy(template.get("watermark", {}))
    wm["enabled"] = True
    wm["type"] = "traceflow"
    return wm


def _disabled_watermark() -> Dict[str, Any]:
    return {"enabled": False}


def _apply_method(config: Dict[str, Any], method: str, template: Mapping[str, Any]) -> None:
    method = method.lower()
    if method == "baseline":
        config["security"] = _identity_security(template)
        config["watermark"] = _disabled_watermark()
    elif method == "keyed":
        config["security"] = _keyed_security(template)
        config["watermark"] = _disabled_watermark()
    elif method == "traceflow_identity":
        config["security"] = _identity_security(template)
        config["watermark"] = _enabled_watermark(template)
    elif method == "traceflow":
        config["security"] = _keyed_security(template)
        config["watermark"] = _enabled_watermark(template)
    else:
        raise ValueError(f"Unknown experiment method: {method}")


def _is_single_file_config(raw: Mapping[str, Any]) -> bool:
    return "traceflow" in raw and "experiments" in raw and "training" in raw


def _entry_run_name(raw: Mapping[str, Any], entry: str, fallback: str) -> str:
    return str(raw.get("entries", {}).get(entry, {}).get("run_name") or fallback)


def _bundle_root(raw: Mapping[str, Any], entry: str, run_name: str, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    configured = Path(str(raw.get("artifact_bundle", {}).get("root", "runs/traceflow")))
    if configured.name == run_name:
        return configured
    return configured / run_name


def _bundle_paths(bundle: Path) -> Dict[str, Path]:
    return {
        "root": bundle,
        "configs": bundle / "configs",
        "logs": bundle / "logs",
        "outputs": bundle / "outputs",
        "checkpoints": bundle / "checkpoints",
        "results": bundle / "results",
        "figures": bundle / "figures",
        "training_figures": bundle / "figures" / "training",
        "paper_figures": bundle / "figures" / "paper",
        "reports": bundle / "reports",
    }


def _ensure_bundle(bundle: Path) -> Dict[str, Path]:
    paths = _bundle_paths(bundle)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_bundle_source(raw: Mapping[str, Any], source_config: str, paths: Mapping[str, Path]) -> None:
    write_yaml(paths["configs"] / "traceflow.source.yml", dict(raw))
    write_yaml(paths["configs"] / "traceflow.source.redacted.yml", redact_secrets(raw))
    (paths["configs"] / "source_path.txt").write_text(str(source_config) + "\n", encoding="utf-8")


def _write_bundle_config(resolved: Dict[str, Any], paths: Mapping[str, Path], name: str = "resolved") -> Path:
    private_path = paths["configs"] / f"{name}.private.yml"
    redacted_path = paths["configs"] / f"{name}.redacted.yml"
    write_yaml(private_path, resolved)
    write_yaml(redacted_path, redact_secrets(resolved))
    if name == "resolved":
        private_alias = paths["configs"] / "resolved.private.yml"
        redacted_alias = paths["configs"] / "resolved.redacted.yml"
        if private_path.resolve() != private_alias.resolve():
            shutil.copy2(private_path, private_alias)
        if redacted_path.resolve() != redacted_alias.resolve():
            shutil.copy2(redacted_path, redacted_alias)
    return private_path


def _point_config_at_bundle(resolved: Dict[str, Any], paths: Mapping[str, Path]) -> None:
    resolved.setdefault("project", {})["output_dir"] = str(paths["outputs"])
    resolved.setdefault("training", {})["checkpoint_dir"] = str(paths["checkpoints"])


def _write_manifest(paths: Mapping[str, Path], *, command: str, run_name: str, status: str) -> None:
    manifest = {
        "status": status,
        "command": command,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle": str(paths["root"]),
        "outputs": str(paths["outputs"]),
        "checkpoints": str(paths["checkpoints"]),
        "results": str(paths["results"]),
        "figures": str(paths["figures"]),
        "reports": str(paths["reports"]),
    }
    with open(paths["reports"] / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _write_readme(paths: Mapping[str, Path], *, title: str) -> None:
    text = f"""# {title}

This directory is the complete downloadable TraceFlow artifact bundle.

Important files:

- `configs/resolved.redacted.yml`: public reproducibility config with secrets redacted.
- `configs/resolved.private.yml`: private resolved config. Do not publish if it contains your secret key.
- `logs/main.log`: detached run log when `--detach` was used.
- `outputs/`: training logs, sample grids, and train reports.
- `checkpoints/`: model checkpoints, including `latest.pt`.
- `results/`: exp01-exp05 metrics and attack outputs.
- `figures/`: training and paper-level visualizations.
- `reports/loss_diagnosis.md`: interpretation of training loss and watermark curves.
- `reports/manifest.json`: machine-readable bundle index.
"""
    (paths["root"] / "README_RUN.md").write_text(text, encoding="utf-8")


def _maybe_archive(paths: Mapping[str, Path], enabled: bool) -> None:
    if not enabled:
        return
    archive = paths["root"].with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(paths["root"], arcname=paths["root"].name)
    print(f"[traceflow] archive: {archive}")


def _quote_cmd(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def _launch_detached(args: argparse.Namespace, entry_cmd: str, run_name: str, raw: Mapping[str, Any]) -> bool:
    if args.dry_run or getattr(args, "foreground", False):
        return False
    detach = bool(args.detach or raw.get("runtime", {}).get("detach_default", False))
    if not detach:
        return False
    bundle = _bundle_root(raw, entry_cmd.replace("-", "_"), run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    cmd = [*_python_module_cmd("scripts.traceflow"), entry_cmd, "--config", args.config, "--bundle-dir", str(bundle), "--foreground"]
    if args.smoke:
        cmd.append("--smoke")
    if args.resume:
        cmd.extend(["--resume", args.resume])
    for override in args.set_overrides:
        cmd.extend(["--set", override])
    if getattr(args, "attack", None):
        cmd.extend(["--attack", args.attack])
    if getattr(args, "attack_steps", None) is not None:
        cmd.extend(["--attack-steps", str(args.attack_steps)])
    if getattr(args, "attacker", None):
        cmd.extend(["--attacker", args.attacker])
    run_sh = paths["root"] / "run.sh"
    run_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONUNBUFFERED=1\ncd " + shlex.quote(str(Path.cwd())) + "\n" + _quote_cmd(cmd) + "\n", encoding="utf-8")
    run_sh.chmod(0o755)
    main_log = paths["logs"] / "main.log"
    with open(main_log, "ab") as log_f:
        proc = subprocess.Popen(["nohup", "bash", str(run_sh)], stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    (paths["root"] / "RUNNING.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    _write_readme(paths, title=f"TraceFlow {entry_cmd} bundle")
    _write_manifest(paths, command=entry_cmd, run_name=run_name, status="running_detached")
    print(f"[traceflow] detached PID: {proc.pid}")
    print(f"[traceflow] bundle: {paths['root']}")
    print(f"[traceflow] log:    {main_log}")
    print(f"[traceflow] watch:  tail -f {main_log}")
    return True


def _single_train_config(raw: Mapping[str, Any], overrides: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = _base_config(raw)
    meta = raw.get("traceflow", {})
    _set_project(resolved, name=meta.get("run_name") or meta.get("name"))
    resolved = apply_overrides(resolved, overrides)
    validate_resolved_config(resolved)
    return resolved, {"run": {"name": resolved.get("project", {}).get("name")}, "postprocess": raw.get("postprocess", {})}


def _single_experiment_config(raw: Mapping[str, Any], exp_id: str, overrides: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    exp = raw.get("experiments", {}).get(exp_id)
    if not exp:
        raise ValueError(f"Config does not define {exp_id}")
    resolved = _base_config(raw)
    _apply_method(resolved, exp.get("method", "traceflow"), raw)
    run_name = exp.get("run_name") or f"{exp_id}_{raw.get('traceflow', {}).get('mode', 'full')}"
    _set_project(resolved, name=run_name)
    if exp.get("overrides"):
        resolved = deep_merge(resolved, exp["overrides"])
    resolved = apply_overrides(resolved, overrides)
    validate_resolved_config(resolved)
    return resolved, {
        "exp_id": exp_id,
        "run_name": run_name,
        "attack": exp.get("attack", "latent"),
        "attack_steps": exp.get("attack_steps"),
        "attacker": exp.get("attacker", "no_key"),
        "enabled": exp.get("enabled", True),
    }


def _entry_overrides(raw: Mapping[str, Any], entry: str) -> Dict[str, Any]:
    entry_cfg = raw.get("entries", {}).get(entry, {})
    return deepcopy(entry_cfg.get("overrides", {}))


def _final_entry_config(raw: Mapping[str, Any], *, method: str, run_name: str, entry: str, overrides: List[str], paths: Mapping[str, Path]) -> Dict[str, Any]:
    resolved = _base_config(raw)
    _apply_method(resolved, method, raw)
    _set_project(resolved, name=run_name)
    resolved = deep_merge(resolved, _entry_overrides(raw, entry))
    _point_config_at_bundle(resolved, paths)
    resolved = apply_overrides(resolved, overrides)
    _point_config_at_bundle(resolved, paths)
    validate_resolved_config(resolved)
    return resolved


def _run_training_plots(run_name: str, output_root: Path | str = "outputs/flow_transformer", output_dir: Optional[Path] = None, dry_run: bool = False) -> int:
    run_dir = Path(output_root) / run_name
    out = output_dir or (run_dir / "figures")
    cmd = [*_python_module_cmd("scripts.plot_training_run"), "--run-dir", str(run_dir), "--output-dir", str(out)]
    return _run(cmd, dry_run=dry_run)


def _run_loss_diagnosis(paths: Mapping[str, Path], dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.analyze_training_loss"), "--run-dir", str(paths["root"]), "--output-dir", str(paths["reports"])]
    return _run(cmd, dry_run=dry_run)


def _sync_training_reports(paths: Mapping[str, Path], source_dir: Path) -> None:
    for name in ("training_summary.csv", "training_summary.md"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, paths["reports"] / name)


def _write_readiness_report(results_dir: Path, reports_dir: Path) -> None:
    from scripts.check_experiment_readiness import check
    warnings = check(results_dir)
    lines = ["# TraceFlow Readiness Report", "", f"Results: `{results_dir}`", "", f"Warnings: `{len(warnings)}`", ""]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("All required experiment outputs and figures are present.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train_final_entry(args: argparse.Namespace, *, entry: str, method: str, fallback_run_name: str) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, entry, fallback_run_name)
    if _launch_detached(args, entry.replace("_", "-"), run_name, raw):
        return 0
    bundle = _bundle_root(raw, entry, run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    _write_readme(paths, title=f"TraceFlow {entry.replace('_', ' ')} bundle")
    resolved = _final_entry_config(raw, method=method, run_name=run_name, entry=entry, overrides=args.set_overrides, paths=paths)
    config_path = _write_bundle_config(resolved, paths)
    print(f"[traceflow] entry:  {entry}")
    print(f"[traceflow] bundle: {paths['root']}")
    print(f"[traceflow] config: {config_path}")
    cmd = [*_python_module_cmd("scripts.train_flow_transformer"), "--config", str(config_path), "--run-name", run_name]
    if args.smoke:
        cmd.append("--smoke")
    if args.resume:
        cmd.extend(["--resume", args.resume])
    rc = _run(cmd, dry_run=args.dry_run, log_path=paths["logs"] / "train.log")
    if rc != 0:
        _write_manifest(paths, command=entry, run_name=run_name, status="error")
        return rc
    fig_dir = paths["training_figures"]
    rc = _run_training_plots(run_name, output_root=paths["outputs"], output_dir=fig_dir, dry_run=args.dry_run)
    if rc != 0:
        return rc
    if not args.dry_run:
        _sync_training_reports(paths, fig_dir)
    rc = _run_loss_diagnosis(paths, dry_run=args.dry_run)
    if rc != 0:
        return rc
    _write_manifest(paths, command=entry, run_name=run_name, status="dry_run" if args.dry_run else "ok")
    _maybe_archive(paths, bool(args.make_archive or raw.get("runtime", {}).get("make_archive", False)) and not args.dry_run)
    print(f"[traceflow] complete bundle: {paths['root']}")
    return 0


def _train_generator(args: argparse.Namespace) -> int:
    return _train_final_entry(args, entry="train_generator", method="baseline", fallback_run_name="traceflow-generator")


def _train_final(args: argparse.Namespace) -> int:
    return _train_final_entry(args, entry="train_final", method="traceflow", fallback_run_name="traceflow-final")


# ---------------------------------------------------------------------------
# Preflight readiness entry
# ---------------------------------------------------------------------------

def _preflight_add(checks: List[Dict[str, Any]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})
    icon = {"ok": "OK", "warn": "WARN", "error": "ERROR"}.get(status, status.upper())
    print(f"[check-ready] {icon:5s} {name}: {detail}")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _count_images(root: Path, *, nested: bool) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if not root.exists():
        return 0
    iterator = root.rglob("*") if nested else root.iterdir()
    return sum(1 for p in iterator if p.is_file() and p.suffix.lower() in exts)


def _check_data(cfg: Mapping[str, Any], checks: List[Dict[str, Any]]) -> None:
    data = cfg.get("data", {})
    name = data.get("name", "random")
    root = Path(str(data.get("root", "./data")))
    if name == "random":
        _preflight_add(checks, "data", "ok", "random smoke data does not require files")
    elif name == "cifar10":
        extracted = root / "cifar-10-batches-py"
        archive = root / "cifar-10-python.tar.gz"
        if extracted.exists():
            _preflight_add(checks, "data.cifar10", "ok", f"found extracted CIFAR-10 at {extracted}")
        elif archive.exists():
            _preflight_add(checks, "data.cifar10", "error", f"found {archive}, but it must be extracted to {extracted}")
        elif data.get("download"):
            _preflight_add(checks, "data.cifar10", "warn", "CIFAR-10 is missing but data.download=true may download it during training")
        else:
            _preflight_add(checks, "data.cifar10", "error", f"missing {extracted}; extract CIFAR-10 or change data.name/root")
    elif name == "imagefolder":
        count = _count_images(root, nested=True)
        class_dirs = [p for p in root.iterdir()] if root.exists() else []
        class_dirs = [p for p in class_dirs if p.is_dir()]
        if count > 0 and class_dirs:
            _preflight_add(checks, "data.imagefolder", "ok", f"found {count} images under {root}")
        elif root.exists():
            _preflight_add(checks, "data.imagefolder", "error", f"{root} exists but has no class-subdir images")
        else:
            _preflight_add(checks, "data.imagefolder", "error", f"missing imagefolder root: {root}")
    elif name == "flat":
        count = _count_images(root, nested=False)
        if count > 0:
            _preflight_add(checks, "data.flat", "ok", f"found {count} images in {root}")
        elif root.exists():
            _preflight_add(checks, "data.flat", "error", f"{root} exists but has no images")
        else:
            _preflight_add(checks, "data.flat", "error", f"missing flat image root: {root}")
    else:
        _preflight_add(checks, "data", "error", f"unknown data.name={name!r}")


def _check_training_budget(cfg: Mapping[str, Any], checks: List[Dict[str, Any]], *, name: str = "training.batch") -> None:
    training = cfg.get("training", {})
    try:
        batch_size = int(training.get("batch_size", 0))
        grad_accum = int(training.get("grad_accum_steps", 1))
        num_steps = int(training.get("num_steps", 0))
    except (TypeError, ValueError):
        _preflight_add(checks, name, "error", "batch_size, grad_accum_steps, and num_steps must be integers")
        return
    effective_batch = batch_size * grad_accum
    detail = f"batch_size={batch_size}, grad_accum_steps={grad_accum}, effective_batch={effective_batch}, num_steps={num_steps}"
    if batch_size <= 0 or grad_accum <= 0 or num_steps <= 0:
        _preflight_add(checks, name, "error", detail + "; values must be positive")
    elif batch_size > 32:
        _preflight_add(checks, name, "warn", detail + "; micro-batch >32 may OOM on full TraceFlow")
    elif batch_size < 8 and str(cfg.get("project", {}).get("device", "auto")) == "cuda":
        _preflight_add(checks, name, "warn", detail + "; safe but likely underuses a 32 GB GPU")
    else:
        _preflight_add(checks, name, "ok", detail)


def _write_preflight_report(paths: Mapping[str, Path], checks: List[Dict[str, Any]], *, strict: bool) -> None:
    errors = [c for c in checks if c["status"] == "error"]
    warnings = [c for c in checks if c["status"] == "warn"]
    status = "error" if errors or (strict and warnings) else "ok"
    payload = {"status": status, "strict": strict, "errors": len(errors), "warnings": len(warnings), "checks": checks}
    with open(paths["reports"] / "preflight_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = ["# TraceFlow Preflight Report", "", f"Status: `{status}`", f"Errors: `{len(errors)}`", f"Warnings: `{len(warnings)}`", "", "| Check | Status | Detail |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    (paths["reports"] / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_ready(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    effective_raw = apply_overrides(dict(raw), args.set_overrides)
    run_name = _entry_run_name(raw, "check_ready", "traceflow-preflight")
    bundle = _bundle_root(raw, "check_ready", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    checks: List[Dict[str, Any]] = []

    _preflight_add(checks, "python", "ok", f"{platform.python_version()} at {sys.executable}")
    _preflight_add(checks, "logging.unbuffered", "ok", "subprocesses use python -u and PYTHONUNBUFFERED=1")
    _preflight_add(checks, "platform", "ok", platform.platform())

    for mod in ("yaml", "PIL", "matplotlib", "numpy"):
        _preflight_add(checks, f"module.{mod}", "ok" if _module_available(mod) else "error", "available" if _module_available(mod) else "missing")
    for mod in ("torch", "torchvision"):
        _preflight_add(checks, f"module.{mod}", "ok" if _module_available(mod) else "error", "available" if _module_available(mod) else "missing")
    if effective_raw.get("autoencoder", {}).get("backend") == "diffusers":
        for mod in ("diffusers", "transformers", "accelerate", "safetensors"):
            _preflight_add(checks, f"module.{mod}", "ok" if _module_available(mod) else "error", "available" if _module_available(mod) else "missing")

    if _module_available("torch"):
        import torch
        device = str(effective_raw.get("project", {}).get("device", "auto"))
        if device == "cuda":
            if torch.cuda.is_available():
                _preflight_add(checks, "cuda", "ok", f"{torch.cuda.get_device_name(0)} | torch CUDA {torch.version.cuda}")
            else:
                _preflight_add(checks, "cuda", "error", "project.device=cuda but torch.cuda.is_available() is false")
        elif device == "auto":
            detail = "CUDA available" if torch.cuda.is_available() else "CUDA unavailable; auto will fall back if supported"
            _preflight_add(checks, "device.auto", "ok", detail)
        else:
            _preflight_add(checks, "device", "ok", f"configured device={device}")

    try:
        resolved_final = _final_entry_config(raw, method="traceflow", run_name=_entry_run_name(raw, "train_final", "traceflow-final"), entry="train_final", overrides=args.set_overrides, paths=paths)
        _write_bundle_config(resolved_final, paths, name="preflight_final")
        _preflight_add(checks, "config.train_final", "ok", "resolved full TraceFlow config validates")
        _check_training_budget(resolved_final, checks, name="training.train_final")
    except Exception as exc:
        resolved_final = _base_config(raw)
        _preflight_add(checks, "config.train_final", "error", str(exc))
    try:
        resolved_gen = _final_entry_config(raw, method="baseline", run_name=_entry_run_name(raw, "train_generator", "traceflow-generator"), entry="train_generator", overrides=args.set_overrides, paths=paths)
        _write_bundle_config(resolved_gen, paths, name="preflight_generator")
        _preflight_add(checks, "config.train_generator", "ok", "resolved generator-only config validates")
        _check_training_budget(resolved_gen, checks, name="training.train_generator")
    except Exception as exc:
        _preflight_add(checks, "config.train_generator", "error", str(exc))
    for exp_id in EXP_ORDER:
        try:
            resolved, _meta = _single_experiment_config(raw, exp_id, args.set_overrides)
            _point_config_at_bundle(resolved, paths)
            validate_resolved_config(resolved)
            _preflight_add(checks, f"config.{exp_id}", "ok", "experiment config validates")
        except Exception as exc:
            _preflight_add(checks, f"config.{exp_id}", "error", str(exc))

    _check_training_budget(effective_raw, checks, name="training.top_level")
    _check_data(effective_raw, checks)

    ae = effective_raw.get("autoencoder", {})
    if ae.get("backend") == "diffusers":
        vae_path = Path(str(ae.get("pretrained_model_name_or_path", "")))
        if vae_path.exists() and (vae_path / "config.json").exists():
            _preflight_add(checks, "autoencoder.vae", "ok", f"found local diffusers VAE at {vae_path}")
        else:
            _preflight_add(checks, "autoencoder.vae", "error", f"missing local diffusers VAE files at {vae_path}")

    secret = effective_raw.get("security", {}).get("latent_transform", {}).get("secret_key", "")
    if "CHANGE_ME" in str(secret) or not secret:
        _preflight_add(checks, "security.secret_key", "warn", "secret_key is placeholder; change it before final paper runs")
    else:
        _preflight_add(checks, "security.secret_key", "ok", "secret_key is set")

    test_file = paths["reports"] / ".write_test"
    try:
        test_file.write_text("ok\n", encoding="utf-8")
        test_file.unlink()
        usage = shutil.disk_usage(paths["root"])
        free_gb = usage.free / (1024 ** 3)
        status = "ok" if free_gb >= 10 else "warn"
        _preflight_add(checks, "bundle.write_and_space", status, f"writable; free space {free_gb:.1f} GB at {paths['root']}")
    except Exception as exc:
        _preflight_add(checks, "bundle.write_and_space", "error", str(exc))

    if shutil.which("nohup"):
        _preflight_add(checks, "detach.nohup", "ok", "nohup is available for --detach")
    else:
        _preflight_add(checks, "detach.nohup", "error", "nohup not found; --detach cannot protect SSH disconnects")

    _write_preflight_report(paths, checks, strict=args.strict)
    _write_manifest(paths, command="check_ready", run_name=run_name, status="checked")
    errors = [c for c in checks if c["status"] == "error"]
    warnings = [c for c in checks if c["status"] == "warn"]
    print(f"[check-ready] report: {paths['reports'] / 'preflight_report.md'}")
    if errors or (args.strict and warnings):
        print(f"[check-ready] not ready: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"[check-ready] ready with {len(warnings)} warning(s)")
    return 0



def _run_all(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, "run_all", "traceflow-paper-all")
    if _launch_detached(args, "run-all", run_name, raw):
        return 0
    bundle = _bundle_root(raw, "run_all", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    _write_readme(paths, title="TraceFlow full experiment bundle")
    single = _is_single_file_config(raw)
    suite_meta = raw.get("traceflow", {}) if single else raw.get("suite", {})
    smoke = args.smoke or suite_meta.get("mode", "full") == "smoke"
    selected = [eid for eid in EXP_ORDER if raw.get("experiments", {}).get(eid, {}).get("enabled", True)] if single else EXP_ORDER
    print(f"[traceflow] entry:       run-all")
    print(f"[traceflow] bundle:      {paths['root']}")
    print(f"[traceflow] mode:        {'smoke' if smoke else 'full'}")
    print(f"[traceflow] experiments: {', '.join(selected)}")
    exp_config_dir = paths["configs"] / "experiments"
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    for exp_id in selected:
        if single:
            resolved, exp_meta = _single_experiment_config(raw, exp_id, args.set_overrides)
        else:
            resolved, exp_meta = compose_suite_experiment(raw, exp_id, repo_root=Path.cwd(), set_overrides=args.set_overrides)
        if exp_meta.get("enabled") is False:
            continue
        resolved = deep_merge(resolved, _entry_overrides(raw, "run_all"))
        resolved = apply_overrides(resolved, args.set_overrides)
        _point_config_at_bundle(resolved, paths)
        validate_resolved_config(resolved)
        config_path = _write_bundle_config(resolved, {**paths, "configs": exp_config_dir}, name=exp_id)
        module_path = REGISTRY[exp_id]
        mod = importlib.import_module(module_path)
        exp_args = argparse.Namespace(
            smoke=smoke,
            dry_run=args.dry_run,
            config=str(config_path),
            resume=args.resume,
            output_dir=str(paths["results"]),
            run_name=exp_meta["run_name"],
            attack=args.attack or exp_meta.get("attack", "latent"),
            steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"),
            attack_steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"),
            attacker=args.attacker or exp_meta.get("attacker", "no_key"),
        )
        print(f"\n[traceflow] running {exp_id} with {config_path}")
        if args.dry_run:
            print(f"  [dry-run] would call {module_path}.run(run_name={exp_args.run_name})")
        else:
            result = mod.run(exp_args)
            if result.get("status") not in ("ok", "dry_run"):
                _write_manifest(paths, command="run_all", run_name=run_name, status="error")
                return 1
        if args.training_figures or suite_meta.get("training_figures"):
            rc = _run_training_plots(exp_meta["run_name"], output_root=paths["outputs"], output_dir=paths["training_figures"] / exp_id, dry_run=args.dry_run)
            if rc != 0:
                return rc
    paper_results_figures = paths["results"] / "figures"
    if args.figures or suite_meta.get("figures", True):
        rc = _figures(argparse.Namespace(results_dir=str(paths["results"]), output_dir=str(paper_results_figures), mode="auto"), dry_run=args.dry_run)
        if rc != 0:
            return rc
        if not args.dry_run and paper_results_figures.exists():
            shutil.copytree(paper_results_figures, paths["paper_figures"], dirs_exist_ok=True)
    if args.readiness or suite_meta.get("readiness", True):
        if args.dry_run:
            print("[traceflow] [dry-run] would run readiness check")
        else:
            _write_readiness_report(paths["results"], paths["reports"])
            if args.strict:
                rc = _readiness(argparse.Namespace(results_dir=str(paths["results"]), strict=True), dry_run=False)
                if rc != 0:
                    return rc
    _write_manifest(paths, command="run_all", run_name=run_name, status="dry_run" if args.dry_run else "ok")
    _maybe_archive(paths, bool(args.make_archive or raw.get("runtime", {}).get("make_archive", False)) and not args.dry_run)
    print(f"[traceflow] complete bundle: {paths['root']}")
    return 0


def _train(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    if _is_single_file_config(raw):
        resolved, meta = _single_train_config(raw, args.set_overrides)
    else:
        resolved, meta = compose_run_config(args.config, repo_root=Path.cwd(), set_overrides=args.set_overrides)
    run_name = resolved.get("project", {}).get("name") or meta.get("run", {}).get("name") or "traceflow-run"
    out_dir = _resolved_dir(run_name)
    config_path = _write_resolved_config(resolved, out_dir / "train.yml")
    print(f"[traceflow] resolved config: {config_path}")
    cmd = [*_python_module_cmd("scripts.train_flow_transformer"), "--config", str(config_path), "--run-name", run_name]
    if args.smoke:
        cmd.append("--smoke")
    if args.resume:
        cmd.extend(["--resume", args.resume])
    rc = _run(cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc
    post = meta.get("postprocess", {})
    if args.figures or post.get("training_figures"):
        rc = _run_training_plots(run_name, dry_run=args.dry_run)
        if rc != 0:
            return rc
    return 0


def _experiment(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    single = _is_single_file_config(raw)
    suite_meta = raw.get("traceflow", {}) if single else raw.get("suite", {})
    suite_name = suite_meta.get("name", Path(args.config).stem)
    smoke = args.smoke or suite_meta.get("mode", "full") == "smoke"
    output_dir = Path(args.output_dir or suite_meta.get("output_dir", "results/traceflow"))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_base = _resolved_dir(suite_name)
    resolved_base.mkdir(parents=True, exist_ok=True)
    write_yaml(resolved_base / "source.redacted.yml", redact_secrets(raw))
    available = [eid for eid in EXP_ORDER if raw.get("experiments", {}).get(eid, {}).get("enabled", True)] if single else [eid for eid in EXP_ORDER if eid in raw.get("experiments", {})]
    selected = available if args.all or not args.only else [x.strip() for x in args.only.split(",") if x.strip()]
    unknown = [x for x in selected if x not in EXP_ORDER]
    if unknown:
        raise SystemExit(f"Unknown experiment ids: {unknown}")
    print(f"[traceflow] config: {args.config}")
    print(f"[traceflow] suite: {suite_name}")
    print(f"[traceflow] mode: {'smoke' if smoke else 'full'}")
    print(f"[traceflow] output-dir: {output_dir}")
    print(f"[traceflow] experiments: {', '.join(selected)}")
    for exp_id in selected:
        if single:
            resolved, exp_meta = _single_experiment_config(raw, exp_id, args.set_overrides)
        else:
            resolved, exp_meta = compose_suite_experiment(raw, exp_id, repo_root=Path.cwd(), set_overrides=args.set_overrides)
        if exp_meta.get("enabled") is False:
            continue
        config_path = _write_resolved_config(resolved, resolved_base / f"{exp_id}.yml")
        module_path = REGISTRY[exp_id]
        mod = importlib.import_module(module_path)
        exp_args = argparse.Namespace(smoke=smoke, dry_run=args.dry_run, config=str(config_path), resume=args.resume, output_dir=str(output_dir), run_name=exp_meta["run_name"], attack=args.attack or exp_meta.get("attack", "latent"), steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"), attack_steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"), attacker=args.attacker or exp_meta.get("attacker", "no_key"))
        print(f"\n[traceflow] running {exp_id} with {config_path}")
        if args.dry_run:
            print(f"  [dry-run] would call {module_path}.run(run_name={exp_args.run_name})")
            continue
        result = mod.run(exp_args)
        if result.get("status") not in ("ok", "dry_run"):
            return 1
        if args.training_figures or suite_meta.get("training_figures"):
            rc = _run_training_plots(exp_meta["run_name"], dry_run=False)
            if rc != 0:
                return rc
    if not args.dry_run and (args.figures or suite_meta.get("figures")):
        rc = _figures(argparse.Namespace(results_dir=str(output_dir), output_dir=str(output_dir / "figures"), mode="auto"), dry_run=False)
        if rc != 0:
            return rc
    elif args.dry_run and (args.figures or suite_meta.get("figures")):
        print("[traceflow] [dry-run] would generate paper figures")
    if not args.dry_run and (args.readiness or suite_meta.get("readiness")):
        rc = _readiness(argparse.Namespace(results_dir=str(output_dir), strict=args.strict), dry_run=False)
        if rc != 0:
            return rc
    elif args.dry_run and (args.readiness or suite_meta.get("readiness")):
        print("[traceflow] [dry-run] would run readiness check")
    return 0


def _figures(args: argparse.Namespace, dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.make_traceflow_figures"), "--results-dir", args.results_dir, "--output-dir", args.output_dir or str(Path(args.results_dir) / "figures"), "--mode", args.mode]
    return _run(cmd, dry_run=dry_run)


def _readiness(args: argparse.Namespace, dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.check_experiment_readiness"), "--results-dir", args.results_dir]
    if args.strict:
        cmd.append("--strict")
    return _run(cmd, dry_run=dry_run)


def _add_common_entry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--detach", action="store_true", help="Launch through nohup and return immediately.")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--make-archive", action="store_true", help="Create <bundle>.tar.gz after a foreground run.")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[], help="Dotted override, e.g. training.num_steps=20000")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TraceFlow single-YAML server-friendly CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check-ready", help="Check environment, data, VAE, CUDA, bundle paths, and configs before training.")
    check.add_argument("--config", default=DEFAULT_CONFIG)
    check.add_argument("--bundle-dir", default=None)
    check.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    check.add_argument("--set", dest="set_overrides", action="append", default=[], help="Dotted override, e.g. data.root=/root/autodl-tmp/images")
    check.set_defaults(func=_check_ready)

    gen = sub.add_parser("train-generator", help="Train only the baseline generative model into one artifact bundle.")
    _add_common_entry_args(gen)
    gen.set_defaults(func=_train_generator)
    final = sub.add_parser("train-final", help="Train the full TraceFlow model into one artifact bundle.")
    _add_common_entry_args(final)
    final.set_defaults(func=_train_final)
    all_exp = sub.add_parser("run-all", help="Run exp01-exp05 and generate all paper artifacts into one bundle.")
    _add_common_entry_args(all_exp)
    all_exp.add_argument("--attack", choices=["latent", "pixel", "both"], default=None)
    all_exp.add_argument("--attack-steps", type=int, default=None)
    all_exp.add_argument("--attacker", choices=["no_key", "oracle_key", "both"], default=None)
    all_exp.add_argument("--figures", action="store_true", default=True)
    all_exp.add_argument("--training-figures", action="store_true", default=True)
    all_exp.add_argument("--readiness", action="store_true", default=True)
    all_exp.add_argument("--strict", action="store_true")
    all_exp.set_defaults(func=_run_all)
    train = sub.add_parser("train", help="Advanced/debug: train one model from configs/traceflow.yml.")
    train.add_argument("--config", default=DEFAULT_CONFIG)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", default=None)
    train.add_argument("--figures", action="store_true")
    train.add_argument("--set", dest="set_overrides", action="append", default=[])
    train.set_defaults(func=_train)
    exp = sub.add_parser("experiment", help="Advanced/debug: run exp01-exp05 without bundle layout.")
    exp.add_argument("--config", default=DEFAULT_CONFIG)
    exp.add_argument("--all", action="store_true")
    exp.add_argument("--only", default=None)
    exp.add_argument("--smoke", action="store_true")
    exp.add_argument("--dry-run", action="store_true")
    exp.add_argument("--resume", default=None)
    exp.add_argument("--output-dir", default=None)
    exp.add_argument("--attack", choices=["latent", "pixel", "both"], default=None)
    exp.add_argument("--attack-steps", type=int, default=None)
    exp.add_argument("--attacker", choices=["no_key", "oracle_key", "both"], default=None)
    exp.add_argument("--figures", action="store_true")
    exp.add_argument("--training-figures", action="store_true")
    exp.add_argument("--readiness", action="store_true")
    exp.add_argument("--strict", action="store_true")
    exp.add_argument("--set", dest="set_overrides", action="append", default=[])
    exp.set_defaults(func=_experiment)
    figs = sub.add_parser("figures", help="Advanced/debug: generate paper figures from a results directory.")
    figs.add_argument("--results-dir", required=True)
    figs.add_argument("--output-dir", default=None)
    figs.add_argument("--mode", default="auto", choices=["auto", "smoke", "full"])
    figs.set_defaults(func=lambda args: _figures(args, dry_run=False))
    ready = sub.add_parser("readiness", help="Advanced/debug: check experiment completeness.")
    ready.add_argument("--results-dir", required=True)
    ready.add_argument("--strict", action="store_true")
    ready.set_defaults(func=lambda args: _readiness(args, dry_run=False))
    return p.parse_args()


if __name__ == "__main__":
    ns = _parse_args()
    raise SystemExit(ns.func(ns))
