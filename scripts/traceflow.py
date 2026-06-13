"""
scripts.traceflow
=================
Unified server-friendly command-line entry for TraceFlow.

Public workflow entries:
  python -m scripts.traceflow train-generator --config configs/traceflow.yml
  python -m scripts.traceflow train-keyed     --config configs/traceflow.yml
  python -m scripts.traceflow train-identity  --config configs/traceflow.yml
  python -m scripts.traceflow train-final     --config configs/traceflow.yml
  python -m scripts.traceflow run-all         --config configs/traceflow.yml
  python -m scripts.traceflow eval-all        --config configs/traceflow.yml
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
import urllib.request
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
BASE_SECTIONS = ["naming", "assets", "hardware", "project", "data", "autoencoder", "model", "training", "sampling", "smoke", "security", "watermark"]
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


def _name_context(raw: Mapping[str, Any]) -> Dict[str, Any]:
    naming = raw.get("naming", {}) or {}
    project = str(naming.get("project") or raw.get("traceflow", {}).get("name") or raw.get("project", {}).get("name") or "traceflow")
    dataset = raw.get("data", {}) or {}
    training = raw.get("training", {}) or {}
    dataset_tag = str(naming.get("dataset_tag") or dataset.get("tag") or dataset.get("name") or "dataset")
    budget_tag = str(naming.get("budget_tag") or training.get("budget_tag") or "run")
    num_steps = int(training.get("num_steps", 0) or 0)
    step_tag = str(naming.get("step_tag") or (f"{num_steps // 1000}k" if num_steps and num_steps % 1000 == 0 else str(num_steps)))
    return {
        "project": project,
        "dataset": dataset_tag,
        "dataset_tag": dataset_tag,
        "budget": budget_tag,
        "budget_tag": budget_tag,
        "num_steps": num_steps,
        "step_tag": step_tag,
    }


def _format_name_template(raw: Mapping[str, Any], template: str) -> str:
    ctx = _name_context(raw)
    try:
        return str(template).format(**ctx)
    except KeyError as exc:
        raise KeyError(f"Unknown naming template key {exc} in template {template!r}") from exc


def _entry_run_name(raw: Mapping[str, Any], entry: str, fallback: str) -> str:
    entry_cfg = raw.get("entries", {}).get(entry, {}) or {}
    if entry_cfg.get("run_name"):
        return str(entry_cfg["run_name"])
    templates = (raw.get("naming", {}) or {}).get("templates", {}) or {}
    if entry in templates:
        return _format_name_template(raw, str(templates[entry]))
    return _format_name_template(raw, fallback) if "{" in str(fallback) else str(fallback)


def _experiment_run_name(raw: Mapping[str, Any], exp_id: str, exp: Mapping[str, Any]) -> str:
    if exp.get("run_name"):
        return str(exp["run_name"])
    templates = (raw.get("naming", {}) or {}).get("templates", {}) or {}
    if exp_id in templates:
        return _format_name_template(raw, str(templates[exp_id]))
    method = str(exp.get("method", exp_id)).replace("_", "-")
    return _format_name_template(raw, "{project}-{dataset}-{budget}-" + exp_id + "-" + method)


def _bundle_root(raw: Mapping[str, Any], entry: str, run_name: str, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    configured_raw = str(raw.get("artifact_bundle", {}).get("root", "runs/traceflow"))
    configured = Path(_format_name_template(raw, configured_raw) if "{" in configured_raw else configured_raw)
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


def _write_checkpoint_manifest(paths: Mapping[str, Path], manifest: Mapping[str, Any]) -> None:
    out = paths["reports"] / "checkpoint_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _checkpoint_registry(raw: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Optional[str]]:
    cfg = raw.get("checkpoints", {}) or {}
    return {
        "generator": getattr(args, "generator_checkpoint", None) or cfg.get("generator"),
        "keyed": getattr(args, "keyed_checkpoint", None) or cfg.get("keyed"),
        "traceflow_identity": getattr(args, "identity_checkpoint", None) or cfg.get("traceflow_identity"),
        "traceflow": getattr(args, "traceflow_checkpoint", None) or cfg.get("traceflow"),
    }


def _checkpoint_key_for_exp(exp_id: str) -> str:
    return {
        "exp01": "generator",
        "exp02": "keyed",
        "exp03": "traceflow_identity",
        "exp04": "traceflow",
        "exp05": "traceflow",
    }[exp_id]


def _required_checkpoint_for_eval(exp_id: str) -> bool:
    return exp_id in {"exp01", "exp04", "exp05"}


def _merged_entry_overrides(raw: Mapping[str, Any], *entries: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for entry in entries:
        merged = deep_merge(merged, _entry_overrides(raw, entry))
    return merged


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
    for attr, opt in (
        ("generator_checkpoint", "--generator-checkpoint"),
        ("traceflow_checkpoint", "--traceflow-checkpoint"),
        ("keyed_checkpoint", "--keyed-checkpoint"),
        ("identity_checkpoint", "--identity-checkpoint"),
    ):
        value = getattr(args, attr, None)
        if value:
            cmd.extend([opt, value])
    if getattr(args, "train_missing", False):
        cmd.append("--train-missing")
    for attr, opt in (
        ("stages", "--stages"),
        ("candidates", "--candidates"),
        ("steps", "--steps"),
        ("target_steps", "--target-steps"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([opt, str(value)])
    if getattr(args, "stop_on_first_oom", False):
        cmd.append("--stop-on-first-oom")
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
    _set_project(resolved, name=_format_name_template(raw, meta.get("run_name") or meta.get("name") or "{project}-{dataset}-{budget}"))
    resolved = apply_overrides(resolved, overrides)
    validate_resolved_config(resolved)
    return resolved, {"run": {"name": resolved.get("project", {}).get("name")}, "postprocess": raw.get("postprocess", {})}


def _single_experiment_config(raw: Mapping[str, Any], exp_id: str, overrides: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    exp = raw.get("experiments", {}).get(exp_id)
    if not exp:
        raise ValueError(f"Config does not define {exp_id}")
    resolved = _base_config(raw)
    _apply_method(resolved, exp.get("method", "traceflow"), raw)
    run_name = _experiment_run_name(raw, exp_id, exp)
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
    return _run_training_plots_for_dir(run_dir, out, dry_run=dry_run)


def _run_training_plots_for_dir(run_dir: Path | str, output_dir: Path | str, dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.plot_training_run"), "--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    return _run(cmd, dry_run=dry_run)


def _run_loss_diagnosis(paths: Mapping[str, Path], dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.analyze_training_loss"), "--run-dir", str(paths["root"]), "--output-dir", str(paths["reports"])]
    return _run(cmd, dry_run=dry_run)


def _run_paper_metrics(config_path: Path | str, paths: Mapping[str, Path], dry_run: bool = False) -> int:
    cmd = [
        *_python_module_cmd("scripts.evaluate_paper_metrics"),
        "--config", str(config_path),
        "--bundle-dir", str(paths["root"]),
        "--output-dir", str(paths["reports"] / "paper_metrics"),
    ]
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


def _train_keyed(args: argparse.Namespace) -> int:
    return _train_final_entry(args, entry="train_keyed", method="keyed", fallback_run_name="traceflow-keyed")


def _train_identity(args: argparse.Namespace) -> int:
    return _train_final_entry(args, entry="train_identity", method="traceflow_identity", fallback_run_name="traceflow-identity")


def _train_final(args: argparse.Namespace) -> int:
    return _train_final_entry(args, entry="train_final", method="traceflow", fallback_run_name="traceflow-final")


RUN_ALL_TRAIN_STAGES = [
    {
        "key": "generator",
        "entry": "train_generator",
        "method": "baseline",
        "fallback": "traceflow-generator",
        "title": "generator baseline",
    },
    {
        "key": "keyed",
        "entry": "train_keyed",
        "method": "keyed",
        "fallback": "traceflow-keyed",
        "title": "exp02 keyed-only",
    },
    {
        "key": "traceflow_identity",
        "entry": "train_identity",
        "method": "traceflow_identity",
        "fallback": "traceflow-identity",
        "title": "exp03 TraceFlow identity",
    },
    {
        "key": "traceflow",
        "entry": "train_final",
        "method": "traceflow",
        "fallback": "traceflow-final",
        "title": "full TraceFlow",
    },
]


def _run_all_alias_checkpoint(paths: Mapping[str, Path], key: str) -> Path:
    alias = "identity" if key == "traceflow_identity" else key
    return paths["checkpoints"] / alias / "latest.pt"


def _run_all_canonical_checkpoint(paths: Mapping[str, Path], run_name: str) -> Path:
    return paths["checkpoints"] / run_name / "latest.pt"


def _copy_checkpoint_to_alias(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    src = Path(src).expanduser()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _log_has_oom(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except FileNotFoundError:
        return False
    needles = (
        "cuda out of memory",
        "outofmemoryerror",
        "out of memory",
        "cublas_status_alloc_failed",
    )
    return any(needle in text for needle in needles)


def _resolve_reusable_checkpoint(
    *,
    raw: Mapping[str, Any],
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    key: str,
    run_name: str,
) -> Optional[Path]:
    """Return an existing checkpoint for run-all reuse, if one is available."""
    canonical = _run_all_canonical_checkpoint(paths, run_name)
    if canonical.exists():
        return canonical
    alias = _run_all_alias_checkpoint(paths, key)
    if alias.exists():
        return alias
    registry = _checkpoint_registry(raw, args)
    configured = registry.get(key)
    if configured and Path(configured).expanduser().exists():
        return Path(configured).expanduser()
    return None


def _train_stage_for_run_all(
    *,
    raw: Mapping[str, Any],
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    stage: Mapping[str, str],
    smoke: bool,
    force_train: bool,
    oom_retry: bool,
    checkpoint_manifest: Dict[str, Any],
) -> Optional[Path]:
    key = stage["key"]
    entry = stage["entry"]
    method = stage["method"]
    run_name = _entry_run_name(raw, entry, stage["fallback"])
    alias = _run_all_alias_checkpoint(paths, key)
    reusable = None if force_train else _resolve_reusable_checkpoint(
        raw=raw,
        args=args,
        paths=paths,
        key=key,
        run_name=run_name,
    )
    if reusable is not None:
        print(f"[traceflow] Stage 1: reuse {stage['title']} checkpoint -> {reusable}")
        _copy_checkpoint_to_alias(reusable, alias, dry_run=args.dry_run)
        checkpoint_manifest["train_stages"][key] = {
            "status": "reused" if not args.dry_run else "dry_run_reuse",
            "run_name": run_name,
            "checkpoint": str(alias if not args.dry_run else reusable),
            "source_checkpoint": str(reusable),
        }
        return alias if not args.dry_run else reusable

    print(f"[traceflow] Stage 1: train {stage['title']} -> {run_name}")
    retry_overrides: List[Dict[str, Any]] = [{}]
    if oom_retry and key in {"traceflow", "traceflow_identity"}:
        retry_overrides.extend([
            {"training": {"batch_size": 12, "grad_accum_steps": 1}},
            {"training": {"batch_size": 8, "grad_accum_steps": 1}},
            {"training": {"batch_size": 4, "grad_accum_steps": 1}},
        ])

    attempts = []
    for attempt_idx, retry_override in enumerate(retry_overrides, start=1):
        resolved = _final_entry_config(
            raw,
            method=method,
            run_name=run_name,
            entry=entry,
            overrides=args.set_overrides,
            paths=paths,
        )
        if retry_override:
            resolved = deep_merge(resolved, retry_override)
            _point_config_at_bundle(resolved, paths)
            validate_resolved_config(resolved)
        config_name = f"train_{key}_attempt{attempt_idx}"
        config_path = _write_bundle_config(resolved, paths, name=config_name)
        log_path = paths["logs"] / f"train_{key}.log"
        if retry_override:
            print(
                f"[traceflow] OOM retry for {key}: "
                f"batch_size={resolved['training']['batch_size']} "
                f"grad_accum_steps={resolved['training']['grad_accum_steps']}"
            )
        cmd = [*_python_module_cmd("scripts.train_flow_transformer"), "--config", str(config_path), "--run-name", run_name]
        if smoke:
            cmd.append("--smoke")
        if args.resume:
            cmd.extend(["--resume", args.resume])
        rc = _run(cmd, dry_run=args.dry_run, log_path=log_path)
        attempts.append({
            "attempt": attempt_idx,
            "returncode": rc,
            "config": str(config_path),
            "batch_size": resolved.get("training", {}).get("batch_size"),
            "grad_accum_steps": resolved.get("training", {}).get("grad_accum_steps"),
        })
        if rc == 0:
            canonical = _run_all_canonical_checkpoint(paths, run_name)
            _copy_checkpoint_to_alias(canonical, alias, dry_run=args.dry_run)
            _run_training_plots(run_name, output_root=paths["outputs"], output_dir=paths["training_figures"] / key, dry_run=args.dry_run)
            checkpoint_manifest["train_stages"][key] = {
                "status": "trained" if not args.dry_run else "dry_run_train",
                "run_name": run_name,
                "checkpoint": str(alias),
                "canonical_checkpoint": str(canonical),
                "attempts": attempts,
            }
            # Return the canonical checkpoint for immediate evaluation so
            # experiment modules can infer the matching outputs/<run_name> dir.
            # The alias is still copied for a stable downloadable bundle API.
            return canonical
        if not (oom_retry and _log_has_oom(log_path) and attempt_idx < len(retry_overrides)):
            break
        print(f"[traceflow] {key} failed with CUDA OOM; retrying with a smaller micro-batch.")

    checkpoint_manifest["train_stages"][key] = {
        "status": "failed",
        "run_name": run_name,
        "checkpoint": str(alias),
        "attempts": attempts,
    }
    return None



# ---------------------------------------------------------------------------
# Asset preparation entry
# ---------------------------------------------------------------------------

def _asset_cfg(raw: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(raw.get("assets", {}) or {})
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("weights_dir", "weights")
    cfg.setdefault("dataset", {})
    primary = cfg["dataset"]
    primary.setdefault("name", "imagenette2-320")
    primary.setdefault("url", "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz")
    primary.setdefault("archive", str(Path(cfg["data_dir"]) / "imagenette2-320.tgz"))
    primary.setdefault("extract_dir", cfg["data_dir"])
    primary.setdefault("root", str(Path(cfg["data_dir"]) / "imagenette2-320"))
    primary.setdefault("train_root", str(Path(cfg["data_dir"]) / "imagenette2-320" / "train"))
    primary.setdefault("min_size_mb", 250)
    cfg.setdefault("extra_datasets", [])
    for ds in cfg["extra_datasets"]:
        name = str(ds.get("name", "dataset"))
        ds.setdefault("archive", str(Path(cfg["data_dir"]) / f"{name}.tgz"))
        ds.setdefault("extract_dir", cfg["data_dir"])
        ds.setdefault("root", str(Path(cfg["data_dir"]) / name))
        ds.setdefault("train_root", str(Path(cfg["data_dir"]) / name / "train"))
        ds.setdefault("min_size_mb", 0)
    cfg.setdefault("combined_dataset", {})
    combined = cfg["combined_dataset"]
    combined.setdefault("enabled", False)
    combined.setdefault("name", "combined_imagefolder")
    combined.setdefault("root", str(Path(cfg["data_dir"]) / str(combined["name"])))
    combined.setdefault("train_root", str(Path(str(combined["root"])) / "train"))
    combined.setdefault("link_mode", "hardlink")
    combined.setdefault(
        "sources",
        [str(primary["train_root"])] + [str(ds["train_root"]) for ds in cfg.get("extra_datasets", [])],
    )
    cfg.setdefault("metric_weights", {})
    mw = cfg["metric_weights"]
    mw.setdefault("torch_home", str(Path(cfg["weights_dir"]) / "torch"))
    mw.setdefault("hf_home", str(Path(cfg["weights_dir"]) / "huggingface"))
    mw.setdefault("enabled", True)
    return cfg


def _tar_gz_is_valid(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with tarfile.open(path, "r:gz") as tf:
            for _member in tf:
                pass
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _download_file(
    url: str,
    dest: Path,
    *,
    dry_run: bool = False,
    min_size_mb: float = 0.0,
    proxy: Optional[str] = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = int(float(min_size_mb or 0.0) * 1024 * 1024)
    if dest.exists() and dest.stat().st_size > 0 and (not min_bytes or dest.stat().st_size >= min_bytes):
        print(f"[assets] exists: {dest} ({dest.stat().st_size / (1024 ** 2):.1f} MB)")
        return
    if dest.exists() and min_bytes and dest.stat().st_size < min_bytes:
        print(
            f"[assets] resume incomplete file: {dest} "
            f"({dest.stat().st_size / (1024 ** 2):.1f} MB < {min_size_mb:.1f} MB)"
        )
    else:
        print(f"[assets] download: {url} -> {dest}")
    if dry_run:
        print("[assets] [dry-run] skipped download")
        return

    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "--fail", "--retry", "8", "--retry-delay", "5", "-C", "-", "-o", str(dest), url]
        if proxy:
            cmd[1:1] = ["--proxy", proxy]
        print("[assets] $ " + " ".join(shlex.quote(c) for c in cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"curl failed with exit code {rc}: {url}")
        return

    # Fallback for minimal Python environments. This has no progress bar and no
    # resume support, so curl is preferred whenever it is available.
    tmp = dest.with_suffix(dest.suffix + ".part")
    opener = urllib.request.build_opener()
    if proxy:
        opener.add_handler(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    with opener.open(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def _extract_tgz(archive: Path, extract_dir: Path, expected_train_root: Path, *, dry_run: bool = False) -> None:
    image_count = _count_images(expected_train_root, nested=True) if expected_train_root.exists() else 0
    if image_count > 0:
        print(f"[assets] dataset ready: {expected_train_root} ({image_count} images)")
        return
    valid, reason = _tar_gz_is_valid(archive)
    if not valid:
        raise RuntimeError(f"dataset archive is incomplete or invalid: {archive} ({reason})")
    dataset_root = expected_train_root.parent
    if dataset_root.exists() and image_count == 0:
        print(f"[assets] remove incomplete extracted dataset: {dataset_root}")
        if not dry_run:
            shutil.rmtree(dataset_root)
    print(f"[assets] extract: {archive} -> {extract_dir}")
    if dry_run:
        print("[assets] [dry-run] skipped extraction")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)


def _imagefolder_class_dirs(train_root: Path) -> List[Path]:
    if not train_root.exists():
        return []
    return sorted(p for p in train_root.iterdir() if p.is_dir())


def _imagefolder_stats(train_root: Path) -> Dict[str, Any]:
    class_dirs = _imagefolder_class_dirs(train_root)
    return {
        "train_root": str(train_root),
        "exists": train_root.exists(),
        "class_count": len(class_dirs),
        "image_count": _count_images(train_root, nested=True),
        "classes": [p.name for p in class_dirs],
    }


def _prepare_single_dataset(ds: Mapping[str, Any], *, dry_run: bool = False, proxy: Optional[str] = None) -> Dict[str, Any]:
    archive = Path(str(ds["archive"]))
    extract_dir = Path(str(ds["extract_dir"]))
    train_root = Path(str(ds["train_root"]))
    _download_file(
        str(ds["url"]),
        archive,
        dry_run=dry_run,
        min_size_mb=float(ds.get("min_size_mb", 0) or 0),
        proxy=proxy,
    )
    if dry_run and not archive.exists():
        print(f"[assets] [dry-run] would extract missing archive after download: {archive}")
    elif archive.exists():
        _extract_tgz(archive, extract_dir, train_root, dry_run=dry_run)
    stats = _imagefolder_stats(train_root)
    print(
        f"[assets] dataset stats: {ds.get('name', train_root.name)} | "
        f"classes={stats['class_count']} images={stats['image_count']} root={train_root}"
    )
    return stats


def _link_or_copy_file(src: Path, dest: Path, *, link_mode: str) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(src, dest)
            return
        except OSError:
            pass
    if link_mode == "symlink":
        try:
            dest.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


def _build_combined_imagefolder(combined: Mapping[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
    train_root = Path(str(combined["train_root"]))
    sources = [Path(str(p)) for p in combined.get("sources", [])]
    link_mode = str(combined.get("link_mode", "hardlink")).lower()
    source_stats = [_imagefolder_stats(src) for src in sources]
    expected_images = sum(int(s["image_count"]) for s in source_stats)
    current = _imagefolder_stats(train_root)
    if current["image_count"] >= expected_images and expected_images > 0:
        print(
            f"[assets] combined dataset ready: {train_root} "
            f"({current['class_count']} classes, {current['image_count']} images)"
        )
        return {"combined": current, "sources": source_stats, "rebuilt": False}
    if expected_images == 0:
        missing = [str(src) for src, stats in zip(sources, source_stats) if int(stats["image_count"]) == 0]
        raise RuntimeError(f"combined dataset has no source images; missing or empty sources: {missing}")
    print(
        f"[assets] build combined dataset: {train_root} from {len(sources)} sources "
        f"({expected_images} images expected, mode={link_mode})"
    )
    if dry_run:
        print("[assets] [dry-run] skipped combined dataset build")
        return {"combined": current, "sources": source_stats, "rebuilt": True, "dry_run": True}
    if train_root.exists():
        shutil.rmtree(train_root)
    train_root.mkdir(parents=True, exist_ok=True)
    for src_root in sources:
        source_tag = src_root.parent.name
        for class_dir in _imagefolder_class_dirs(src_root):
            dest_class = train_root / class_dir.name
            for src in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}):
                rel = src.relative_to(class_dir)
                dest = dest_class / rel
                if dest.exists():
                    dest = dest_class / f"{source_tag}_{src.name}"
                _link_or_copy_file(src, dest, link_mode=link_mode)
    final = _imagefolder_stats(train_root)
    print(
        f"[assets] combined dataset built: {train_root} "
        f"({final['class_count']} classes, {final['image_count']} images)"
    )
    return {"combined": final, "sources": source_stats, "rebuilt": True}


def _prepare_metric_weight_cache(weights_dir: Path, torch_home: Path, hf_home: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home.resolve())
    report: Dict[str, Any] = {
        "weights_dir": str(weights_dir),
        "torch_home": str(torch_home),
        "hf_home": str(hf_home),
        "lpips": "not_checked",
        "torchmetrics": "not_checked",
    }
    if dry_run:
        print(f"[assets] [dry-run] would prewarm metric weights in {weights_dir}")
        return report
    try:
        import lpips  # type: ignore
        _ = lpips.LPIPS(net="alex")
        report["lpips"] = "ok"
        print("[assets] LPIPS alex weights ready")
    except Exception as exc:
        report["lpips"] = f"warning: {exc}"
        print(f"[assets] warning: LPIPS weight warmup failed: {exc}")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
        from torchmetrics.image.kid import KernelInceptionDistance  # type: ignore
        from torchmetrics.image.inception import InceptionScore  # type: ignore
        _ = FrechetInceptionDistance(feature=2048)
        _ = KernelInceptionDistance(subset_size=10)
        _ = InceptionScore()
        report["torchmetrics"] = "ok"
        print("[assets] TorchMetrics Inception/FID/KID weights ready")
    except Exception as exc:
        report["torchmetrics"] = f"warning: {exc}"
        print(f"[assets] warning: TorchMetrics metric warmup failed: {exc}")
    return report


def _prepare_assets(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    raw = apply_overrides(dict(raw), args.set_overrides)
    cfg = _asset_cfg(raw)
    data_dir = Path(str(cfg["data_dir"]))
    weights_dir = Path(str(cfg["weights_dir"]))
    primary_ds = cfg["dataset"]
    combined = cfg.get("combined_dataset", {}) or {}
    final_train_root = Path(str(combined.get("train_root") if combined.get("enabled") else primary_ds["train_root"]))
    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "weights_dir": str(weights_dir),
        "dataset": dict(primary_ds),
        "extra_datasets": [dict(ds) for ds in cfg.get("extra_datasets", [])],
        "combined_dataset": dict(combined),
        "metric_weights": cfg.get("metric_weights", {}),
    }
    if not args.skip_data:
        dataset_reports = []
        proxy = args.proxy or cfg.get("proxy")
        dataset_reports.append(_prepare_single_dataset(primary_ds, dry_run=args.dry_run, proxy=proxy))
        for extra_ds in cfg.get("extra_datasets", []) or []:
            dataset_reports.append(_prepare_single_dataset(extra_ds, dry_run=args.dry_run, proxy=proxy))
        report["dataset_reports"] = dataset_reports
        if combined.get("enabled"):
            report["combined_report"] = _build_combined_imagefolder(combined, dry_run=args.dry_run)
    else:
        print("[assets] dataset download skipped")
    if not args.skip_metric_weights:
        mw = cfg.get("metric_weights", {}) or {}
        metric_report = _prepare_metric_weight_cache(
            weights_dir,
            Path(str(mw.get("torch_home", weights_dir / "torch"))),
            Path(str(mw.get("hf_home", weights_dir / "huggingface"))),
            dry_run=args.dry_run,
        )
        report["metric_weight_report"] = metric_report
    else:
        print("[assets] metric weight warmup skipped")
    if not args.skip_vae:
        vae_path = Path(str(raw.get("autoencoder", {}).get("pretrained_model_name_or_path", "pretrained/sd-vae-ft-mse")))
        ok = (vae_path / "config.json").exists() and ((vae_path / "diffusion_pytorch_model.safetensors").exists() or (vae_path / "diffusion_pytorch_model.bin").exists())
        report["vae"] = {"path": str(vae_path), "ready": ok}
        print(f"[assets] VAE {'ready' if ok else 'missing'}: {vae_path}")
    reports_dir = Path(str(args.report_dir or "reports/assets"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "asset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# TraceFlow Asset Report",
        "",
        f"- Dataset train root: `{final_train_root}`",
        f"- Primary archive: `{primary_ds['archive']}`",
        f"- Extra datasets: `{len(cfg.get('extra_datasets', []) or [])}`",
        f"- Metric weights: `{weights_dir}`",
        f"- Report: `{reports_dir / 'asset_report.json'}`",
    ]
    (reports_dir / "asset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[assets] report: {reports_dir / 'asset_report.md'}")
    return 0

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
            class_count = len(class_dirs)
            sample_classes = ", ".join(p.name for p in sorted(class_dirs)[:5])
            status = "ok"
            notes = []
            if class_count < 1000:
                status = "warn"
                notes.append("fewer than 1000 classes; this is not full ImageNet-1K")
            if count < 50000:
                status = "warn"
                notes.append("fewer than 50k images; generation quality may be limited")
            detail = f"found {count} images across {class_count} class dirs under {root}; examples: {sample_classes}"
            if notes:
                detail += "; " + "; ".join(notes)
            _preflight_add(checks, "data.imagefolder", status, detail)
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
    hardware_profile = str(cfg.get("hardware", {}).get("profile", "")).lower()
    micro_batch_warn = 128 if "pro_6000" in hardware_profile or "96gb" in hardware_profile else 64
    if batch_size <= 0 or grad_accum <= 0 or num_steps <= 0:
        _preflight_add(checks, name, "error", detail + "; values must be positive")
    elif batch_size > micro_batch_warn:
        _preflight_add(checks, name, "warn", detail + f"; micro-batch >{micro_batch_warn} may OOM for hardware.profile={hardware_profile or 'unspecified'}")
    elif batch_size < 16 and str(cfg.get("project", {}).get("device", "auto")) == "cuda":
        _preflight_add(checks, name, "warn", detail + "; safe but likely underuses high-memory GPUs")
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
    for mod in ("skimage", "lpips", "torchmetrics", "torch_fidelity"):
        _preflight_add(checks, f"module.{mod}", "ok" if _module_available(mod) else "warn", "available" if _module_available(mod) else "missing; paper metric will be skipped or degraded")
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
    try:
        resolved_keyed = _final_entry_config(raw, method="keyed", run_name=_entry_run_name(raw, "train_keyed", "traceflow-keyed"), entry="train_keyed", overrides=args.set_overrides, paths=paths)
        _write_bundle_config(resolved_keyed, paths, name="preflight_keyed")
        _preflight_add(checks, "config.train_keyed", "ok", "resolved keyed-only config validates")
        _check_training_budget(resolved_keyed, checks, name="training.train_keyed")
    except Exception as exc:
        _preflight_add(checks, "config.train_keyed", "error", str(exc))
    try:
        resolved_identity = _final_entry_config(raw, method="traceflow_identity", run_name=_entry_run_name(raw, "train_identity", "traceflow-identity"), entry="train_identity", overrides=args.set_overrides, paths=paths)
        _write_bundle_config(resolved_identity, paths, name="preflight_identity")
        _preflight_add(checks, "config.train_identity", "ok", "resolved TraceFlow identity config validates")
        _check_training_budget(resolved_identity, checks, name="training.train_identity")
    except Exception as exc:
        _preflight_add(checks, "config.train_identity", "error", str(exc))
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
    asset_cfg = _asset_cfg(effective_raw)
    weights_dir = Path(str(asset_cfg.get("weights_dir", "weights")))
    if weights_dir.exists():
        _preflight_add(checks, "assets.weights_dir", "ok", f"found metric/cache weight directory at {weights_dir}")
    else:
        _preflight_add(checks, "assets.weights_dir", "warn", f"missing {weights_dir}; run prepare-assets to prewarm FID/LPIPS caches")

    ae = effective_raw.get("autoencoder", {})
    if ae.get("backend") == "diffusers":
        vae_path = Path(str(ae.get("pretrained_model_name_or_path", "")))
        config_ok = (vae_path / "config.json").exists()
        safetensors = vae_path / "diffusion_pytorch_model.safetensors"
        bin_file = vae_path / "diffusion_pytorch_model.bin"
        weight_file = safetensors if safetensors.exists() else bin_file
        weight_ok = weight_file.exists()
        size_ok = weight_ok and weight_file.stat().st_size > 100 * 1024 * 1024
        if vae_path.exists() and config_ok and size_ok:
            size_mb = weight_file.stat().st_size / (1024 ** 2)
            _preflight_add(checks, "autoencoder.vae", "ok", f"found local diffusers VAE at {vae_path} ({weight_file.name}, {size_mb:.1f} MB)")
        else:
            missing = []
            if not config_ok:
                missing.append("config.json")
            if not weight_ok:
                missing.append("diffusion_pytorch_model.safetensors or .bin")
            elif not size_ok:
                missing.append(f"complete VAE weight file; found too-small {weight_file.name}")
            detail = ", ".join(missing) if missing else "local diffusers VAE files"
            _preflight_add(checks, "autoencoder.vae", "error", f"missing {detail} at {vae_path}")

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



def _eval_all(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, "eval_all", "traceflow-paper-eval")
    if _launch_detached(args, "eval-all", run_name, raw):
        return 0
    bundle = _bundle_root(raw, "eval_all", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    _write_readme(paths, title="TraceFlow checkpoint-reuse evaluation bundle")
    single = _is_single_file_config(raw)
    suite_meta = raw.get("traceflow", {}) if single else raw.get("suite", {})
    smoke = args.smoke or suite_meta.get("mode", "full") == "smoke"
    selected = [eid for eid in EXP_ORDER if raw.get("experiments", {}).get(eid, {}).get("enabled", True)] if single else EXP_ORDER
    registry = _checkpoint_registry(raw, args)
    eval_cfg = raw.get("evaluation", {}) or {}
    train_missing = bool(args.train_missing or eval_cfg.get("train_missing", False))
    allow_skipped = bool(eval_cfg.get("allow_skipped_ablations", True))
    require_traceflow = bool(eval_cfg.get("require_traceflow_checkpoint", True))
    default_attack_steps = args.attack_steps if args.attack_steps is not None else eval_cfg.get("attack_steps")
    print(f"[traceflow] entry:       eval-all")
    print(f"[traceflow] bundle:      {paths['root']}")
    print(f"[traceflow] mode:        {'smoke' if smoke else 'full'}")
    print(f"[traceflow] train-missing: {train_missing}")
    print(f"[traceflow] experiments: {', '.join(selected)}")

    exp_config_dir = paths["configs"] / "experiments"
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest: Dict[str, Any] = {
        "mode": "eval_all",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_missing": train_missing,
        "checkpoints": registry,
        "experiments": {},
    }

    for exp_id in selected:
        if single:
            resolved, exp_meta = _single_experiment_config(raw, exp_id, args.set_overrides)
        else:
            resolved, exp_meta = compose_suite_experiment(raw, exp_id, repo_root=Path.cwd(), set_overrides=args.set_overrides)
        if exp_meta.get("enabled") is False:
            continue
        resolved = deep_merge(resolved, _merged_entry_overrides(raw, "run_all", "eval_all"))
        resolved = apply_overrides(resolved, args.set_overrides)
        _point_config_at_bundle(resolved, paths)
        validate_resolved_config(resolved)
        config_path = _write_bundle_config(resolved, {**paths, "configs": exp_config_dir}, name=exp_id)
        ckpt_key = _checkpoint_key_for_exp(exp_id)
        ckpt = registry.get(ckpt_key)
        ckpt_exists = bool(ckpt and Path(ckpt).expanduser().exists())
        required = _required_checkpoint_for_eval(exp_id)
        if args.dry_run and ckpt:
            ckpt_exists = True
        if not ckpt_exists and required and not train_missing:
            if exp_id in {"exp04", "exp05"} and require_traceflow:
                checkpoint_manifest["experiments"][exp_id] = {
                    "status": "failed_missing_required_checkpoint",
                    "checkpoint_key": ckpt_key,
                    "checkpoint": ckpt,
                }
                _write_checkpoint_manifest(paths, checkpoint_manifest)
                print(f"[traceflow] error: {exp_id} requires --{ckpt_key.replace('_', '-')}-checkpoint or --train-missing")
                _write_manifest(paths, command="eval_all", run_name=run_name, status="error")
                return 1
        if not ckpt_exists and not required and not train_missing and not allow_skipped:
            print(f"[traceflow] error: {exp_id} checkpoint missing and skipped ablations are disabled")
            _write_manifest(paths, command="eval_all", run_name=run_name, status="error")
            return 1
        train_policy = "missing" if train_missing else "never"
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
            steps=default_attack_steps if default_attack_steps is not None else exp_meta.get("attack_steps"),
            attack_steps=default_attack_steps if default_attack_steps is not None else exp_meta.get("attack_steps"),
            attacker=args.attacker or exp_meta.get("attacker", "no_key"),
            checkpoint=ckpt,
            checkpoint_override=ckpt,
            train_policy=train_policy,
        )
        checkpoint_manifest["experiments"][exp_id] = {
            "checkpoint_key": ckpt_key,
            "checkpoint": ckpt,
            "required": required,
            "train_policy": train_policy,
            "status": "dry_run" if args.dry_run else "pending",
        }
        print(f"\n[traceflow] evaluating {exp_id} with {config_path}")
        if ckpt:
            print(f"[traceflow] {exp_id} checkpoint: {ckpt}")
        if args.dry_run:
            print(f"  [dry-run] would call {module_path}.run(run_name={exp_args.run_name}, train_policy={train_policy})")
            continue
        result = mod.run(exp_args)
        status = result.get("status")
        checkpoint_manifest["experiments"][exp_id]["status"] = status
        if status in ("error", "failed"):
            _write_checkpoint_manifest(paths, checkpoint_manifest)
            _write_manifest(paths, command="eval_all", run_name=run_name, status="error")
            return 1
        if args.training_figures or suite_meta.get("training_figures"):
            metrics = result.get("metrics", {})
            run_dir = metrics.get("output_dir")
            if run_dir:
                rc = _run_training_plots_for_dir(Path(run_dir), paths["training_figures"] / exp_id, dry_run=False)
            else:
                rc = _run_training_plots(exp_meta["run_name"], output_root=paths["outputs"], output_dir=paths["training_figures"] / exp_id, dry_run=False)
            if rc != 0:
                print(f"[traceflow] warning: training figure generation failed for {exp_id}; continuing")

    _write_checkpoint_manifest(paths, checkpoint_manifest)
    paper_results_figures = paths["results"] / "figures"
    if args.figures or suite_meta.get("figures", True):
        rc = _figures(argparse.Namespace(results_dir=str(paths["results"]), output_dir=str(paper_results_figures), mode="auto"), dry_run=args.dry_run)
        if rc != 0:
            return rc
        if not args.dry_run and paper_results_figures.exists():
            shutil.copytree(paper_results_figures, paths["paper_figures"], dirs_exist_ok=True)
    metrics_config = paths["configs"] / "resolved.private.yml"
    if not metrics_config.exists():
        metrics_config = Path(args.config)
    rc = _run_paper_metrics(metrics_config, paths, dry_run=args.dry_run)
    if rc != 0:
        print("[traceflow] warning: paper metrics generation failed; continuing")

    if args.readiness or suite_meta.get("readiness", True):
        if args.dry_run:
            print("[traceflow] [dry-run] would run readiness check")
        else:
            _write_readiness_report(paths["results"], paths["reports"])
            if args.strict:
                rc = _readiness(argparse.Namespace(results_dir=str(paths["results"]), strict=True), dry_run=False)
                if rc != 0:
                    return rc
    _write_manifest(paths, command="eval_all", run_name=run_name, status="dry_run" if args.dry_run else "ok")
    _maybe_archive(paths, bool(args.make_archive or raw.get("runtime", {}).get("make_archive", False)) and not args.dry_run)
    print(f"[traceflow] complete bundle: {paths['root']}")
    return 0


def _run_all(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, "run_all", "traceflow-paper-all")
    if _launch_detached(args, "run-all", run_name, raw):
        return 0

    bundle = _bundle_root(raw, "run_all", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    _write_readme(paths, title="TraceFlow RTX PRO 6000 one-click paper bundle")

    single = _is_single_file_config(raw)
    suite_meta = raw.get("traceflow", {}) if single else raw.get("suite", {})
    run_all_cfg = raw.get("run_all", {}) or {}
    smoke = args.smoke or suite_meta.get("mode", "full") == "smoke"
    selected = [eid for eid in EXP_ORDER if raw.get("experiments", {}).get(eid, {}).get("enabled", True)] if single else EXP_ORDER
    force_train = bool(
        args.force_train
        or run_all_cfg.get("force_train", False)
        or str(run_all_cfg.get("train_policy", "missing")).lower() == "always"
    )
    oom_retry = bool(run_all_cfg.get("oom_retry", True) and not args.no_oom_retry)
    diagnose = bool(run_all_cfg.get("diagnose_data", True) and not args.skip_diagnose)

    print(f"[traceflow] entry:       run-all")
    print(f"[traceflow] bundle:      {paths['root']}")
    print(f"[traceflow] mode:        {'smoke' if smoke else 'full'}")
    print(f"[traceflow] hardware:    {raw.get('hardware', {}).get('profile', 'unspecified')}")
    print(f"[traceflow] force-train: {force_train}")
    print(f"[traceflow] oom-retry:   {oom_retry}")
    print(f"[traceflow] experiments: {', '.join(selected)}")

    checkpoint_manifest: Dict[str, Any] = {
        "mode": "run_all_pipeline",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle": str(paths["root"]),
        "force_train": force_train,
        "oom_retry": oom_retry,
        "train_stages": {},
        "experiments": {},
    }

    # Stage 0: dataset/VAE diagnosis. It is intentionally advisory; a failure
    # stops the pipeline because it usually means the expensive training would
    # fail or produce unusable samples.
    if diagnose:
        print("\n[traceflow] Stage 0: dataset/VAE diagnosis")
        diag_args = argparse.Namespace(
            config=args.config,
            bundle_dir=str(paths["root"]),
            checkpoint=None,
            num_samples=8,
            dry_run=args.dry_run,
            set_overrides=args.set_overrides,
        )
        rc = _diagnose_data(diag_args)
        if rc != 0:
            _write_checkpoint_manifest(paths, checkpoint_manifest)
            _write_manifest(paths, command="run_all", run_name=run_name, status="error")
            return rc
    else:
        print("\n[traceflow] Stage 0: dataset/VAE diagnosis skipped")

    # Stage 1: train or reuse exactly the four checkpoints needed by the paper
    # matrix. exp04 and exp05 share the full TraceFlow checkpoint.
    print("\n[traceflow] Stage 1: train/reuse required checkpoints")
    stage_checkpoints: Dict[str, Path] = {}
    for stage in RUN_ALL_TRAIN_STAGES:
        ckpt = _train_stage_for_run_all(
            raw=raw,
            args=args,
            paths=paths,
            stage=stage,
            smoke=smoke,
            force_train=force_train,
            oom_retry=oom_retry,
            checkpoint_manifest=checkpoint_manifest,
        )
        _write_checkpoint_manifest(paths, checkpoint_manifest)
        if ckpt is None:
            _write_manifest(paths, command="run_all", run_name=run_name, status="error")
            return 1
        stage_checkpoints[stage["key"]] = ckpt

    # Stage 2: evaluate exp01-exp05 from the checkpoints produced/reused above.
    # No experiment module is allowed to train in this phase.
    print("\n[traceflow] Stage 2: checkpoint-reuse evaluation")
    exp_config_dir = paths["configs"] / "experiments"
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    for exp_id in selected:
        if single:
            resolved, exp_meta = _single_experiment_config(raw, exp_id, args.set_overrides)
        else:
            resolved, exp_meta = compose_suite_experiment(raw, exp_id, repo_root=Path.cwd(), set_overrides=args.set_overrides)
        if exp_meta.get("enabled") is False:
            continue
        resolved = apply_overrides(resolved, args.set_overrides)
        _point_config_at_bundle(resolved, paths)
        validate_resolved_config(resolved)
        config_path = _write_bundle_config(resolved, {**paths, "configs": exp_config_dir}, name=exp_id)
        ckpt_key = _checkpoint_key_for_exp(exp_id)
        ckpt = stage_checkpoints[ckpt_key]
        module_path = REGISTRY[exp_id]
        mod = importlib.import_module(module_path)
        exp_args = argparse.Namespace(
            smoke=smoke,
            dry_run=args.dry_run,
            config=str(config_path),
            resume=None,
            output_dir=str(paths["results"]),
            run_name=exp_meta["run_name"],
            attack=args.attack or exp_meta.get("attack", "latent"),
            steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"),
            attack_steps=args.attack_steps if args.attack_steps is not None else exp_meta.get("attack_steps"),
            attacker=args.attacker or exp_meta.get("attacker", "no_key"),
            checkpoint=str(ckpt),
            checkpoint_override=str(ckpt),
            train_policy="never",
        )
        checkpoint_manifest["experiments"][exp_id] = {
            "status": "dry_run" if args.dry_run else "pending",
            "checkpoint_key": ckpt_key,
            "checkpoint": str(ckpt),
            "train_policy": "never",
            "run_name": exp_meta["run_name"],
        }
        print(f"\n[traceflow] evaluating {exp_id} with checkpoint {ckpt}")
        if args.dry_run:
            print(f"  [dry-run] would call {module_path}.run(run_name={exp_args.run_name}, train_policy=never)")
            continue
        result = mod.run(exp_args)
        status = result.get("status")
        checkpoint_manifest["experiments"][exp_id]["status"] = status
        if status in ("error", "failed"):
            _write_checkpoint_manifest(paths, checkpoint_manifest)
            _write_manifest(paths, command="run_all", run_name=run_name, status="error")
            return 1

    _write_checkpoint_manifest(paths, checkpoint_manifest)

    # Stage 3: figures, readiness, reports.
    print("\n[traceflow] Stage 3: paper figures and readiness")
    paper_results_figures = paths["results"] / "figures"
    if args.figures or suite_meta.get("figures", True):
        rc = _figures(argparse.Namespace(results_dir=str(paths["results"]), output_dir=str(paper_results_figures), mode="auto"), dry_run=args.dry_run)
        if rc != 0:
            return rc
        if not args.dry_run and paper_results_figures.exists():
            shutil.copytree(paper_results_figures, paths["paper_figures"], dirs_exist_ok=True)
    metrics_config = paths["configs"] / "experiments" / "exp04.private.yml"
    if not metrics_config.exists():
        metrics_config = paths["configs"] / "resolved.private.yml"
    if not metrics_config.exists():
        metrics_config = Path(args.config)
    rc = _run_paper_metrics(metrics_config, paths, dry_run=args.dry_run)
    if rc != 0:
        print("[traceflow] warning: paper metrics generation failed; continuing")

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


def _benchmark_pro6000(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, "benchmark_pro6000", "traceflow-pro6000-benchmark")
    if _launch_detached(args, "benchmark-pro6000", run_name, raw):
        return 0
    bundle = _bundle_root(raw, "benchmark_pro6000", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    _write_readme(paths, title="TraceFlow RTX PRO 6000 benchmark bundle")
    cmd = [
        *_python_module_cmd("scripts.benchmark_pro6000"),
        "--config", args.config,
        "--bundle-dir", str(bundle),
        "--stages", args.stages,
        "--steps", str(args.steps),
    ]
    if args.candidates:
        cmd.extend(["--candidates", args.candidates])
    if args.target_steps is not None:
        cmd.extend(["--target-steps", str(args.target_steps)])
    if args.stop_on_first_oom:
        cmd.append("--stop-on-first-oom")
    for override in args.set_overrides:
        cmd.extend(["--set", override])
    rc = _run(cmd, dry_run=args.dry_run, log_path=paths["logs"] / "benchmark_pro6000.log")
    _write_manifest(paths, command="benchmark-pro6000", run_name=run_name, status="dry_run" if args.dry_run else ("ok" if rc == 0 else "error"))
    return rc


def _figures(args: argparse.Namespace, dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.make_traceflow_figures"), "--results-dir", args.results_dir, "--output-dir", args.output_dir or str(Path(args.results_dir) / "figures"), "--mode", args.mode]
    return _run(cmd, dry_run=dry_run)


def _readiness(args: argparse.Namespace, dry_run: bool = False) -> int:
    cmd = [*_python_module_cmd("scripts.check_experiment_readiness"), "--results-dir", args.results_dir]
    if args.strict:
        cmd.append("--strict")
    return _run(cmd, dry_run=dry_run)


def _diagnose_data(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    run_name = _entry_run_name(raw, "diagnose_data", "traceflow-data-diagnosis")
    bundle = _bundle_root(raw, "diagnose_data", run_name, args.bundle_dir)
    paths = _ensure_bundle(bundle)
    _write_bundle_source(raw, args.config, paths)
    output_dir = paths["reports"] / "data_diagnosis"
    cmd = [
        *_python_module_cmd("scripts.diagnose_generation_data"),
        "--config", args.config,
        "--output-dir", str(output_dir),
    ]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.num_samples is not None:
        cmd.extend(["--num-samples", str(args.num_samples)])
    for override in args.set_overrides:
        cmd.extend(["--set", override])
    rc = _run(cmd, dry_run=args.dry_run, log_path=paths["logs"] / "diagnose_data.log")
    _write_manifest(paths, command="diagnose_data", run_name=run_name, status="dry_run" if args.dry_run else ("ok" if rc == 0 else "error"))
    return rc


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
    assets = sub.add_parser("prepare-assets", help="Download project datasets, build the configured merged ImageFolder, and prewarm FID/LPIPS metric weights into data/ and weights/.")
    assets.add_argument("--config", default=DEFAULT_CONFIG)
    assets.add_argument("--report-dir", default=None)
    assets.add_argument("--skip-data", action="store_true")
    assets.add_argument("--skip-metric-weights", action="store_true")
    assets.add_argument("--skip-vae", action="store_true")
    assets.add_argument("--proxy", default=None, help="Optional proxy, e.g. http://127.0.0.1:7890. curl also respects HTTPS_PROXY/ALL_PROXY.")
    assets.add_argument("--dry-run", action="store_true")
    assets.add_argument("--set", dest="set_overrides", action="append", default=[])
    assets.set_defaults(func=_prepare_assets)
    check = sub.add_parser("check-ready", help="Check environment, data, VAE, CUDA, bundle paths, and configs before training.")
    check.add_argument("--config", default=DEFAULT_CONFIG)
    check.add_argument("--bundle-dir", default=None)
    check.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    check.add_argument("--set", dest="set_overrides", action="append", default=[], help="Dotted override, e.g. data.root=/root/autodl-tmp/images")
    check.set_defaults(func=_check_ready)
    diag = sub.add_parser("diagnose-data", help="Generate real/reconstruction/sample grids to diagnose ImageFolder/VAE/generator quality.")
    diag.add_argument("--config", default=DEFAULT_CONFIG)
    diag.add_argument("--bundle-dir", default=None)
    diag.add_argument("--checkpoint", default=None, help="Optional generator checkpoint for Euler/Heun sample comparison.")
    diag.add_argument("--num-samples", type=int, default=8)
    diag.add_argument("--dry-run", action="store_true")
    diag.add_argument("--set", dest="set_overrides", action="append", default=[], help="Dotted override, e.g. data.root=/root/autodl-tmp/datasets/imagenet/train")
    diag.set_defaults(func=_diagnose_data)


    bench = sub.add_parser("benchmark-pro6000", help="Stress-test RTX PRO 6000 batch sizes and estimate TraceFlow runtime.")
    _add_common_entry_args(bench)
    bench.add_argument("--stages", default="all", help="all or comma list: generator,keyed,identity,traceflow")
    bench.add_argument("--candidates", default=None, help="Override candidates for all stages, e.g. 32x2,48x1,64x1")
    bench.add_argument("--steps", type=int, default=300, help="Short training steps per trial")
    bench.add_argument("--target-steps", type=int, default=None, help="Target steps for runtime estimates; defaults to config training.num_steps")
    bench.add_argument("--stop-on-first-oom", action="store_true")
    bench.set_defaults(func=_benchmark_pro6000)

    gen = sub.add_parser("train-generator", help="Train only the baseline generative model into one artifact bundle.")
    _add_common_entry_args(gen)
    gen.set_defaults(func=_train_generator)
    keyed = sub.add_parser("train-keyed", help="Train the exp02 keyed-only ablation into one artifact bundle.")
    _add_common_entry_args(keyed)
    keyed.set_defaults(func=_train_keyed)
    identity = sub.add_parser("train-identity", help="Train the exp03 TraceFlow identity ablation into one artifact bundle.")
    _add_common_entry_args(identity)
    identity.set_defaults(func=_train_identity)
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
    all_exp.add_argument("--force-train", action="store_true", help="Ignore reusable checkpoints and retrain all run-all stages.")
    all_exp.add_argument("--skip-diagnose", action="store_true", help="Skip dataset/VAE diagnosis stage.")
    all_exp.add_argument("--no-oom-retry", action="store_true", help="Disable automatic full TraceFlow micro-batch fallback on CUDA OOM.")
    all_exp.set_defaults(func=_run_all)
    eval_all = sub.add_parser("eval-all", help="Reuse trained checkpoints for exp01-exp05, figures, and readiness without retraining by default.")
    _add_common_entry_args(eval_all)
    eval_all.add_argument("--generator-checkpoint", default=None)
    eval_all.add_argument("--traceflow-checkpoint", default=None)
    eval_all.add_argument("--keyed-checkpoint", default=None)
    eval_all.add_argument("--identity-checkpoint", default=None)
    eval_all.add_argument("--train-missing", action="store_true", help="Train only experiments whose checkpoint is missing.")
    eval_all.add_argument("--attack", choices=["latent", "pixel", "both"], default=None)
    eval_all.add_argument("--attack-steps", type=int, default=None)
    eval_all.add_argument("--attacker", choices=["no_key", "oracle_key", "both"], default=None)
    eval_all.add_argument("--figures", action="store_true", default=True)
    eval_all.add_argument("--training-figures", action="store_true", default=True)
    eval_all.add_argument("--readiness", action="store_true", default=True)
    eval_all.add_argument("--strict", action="store_true")
    eval_all.set_defaults(func=_eval_all)
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
