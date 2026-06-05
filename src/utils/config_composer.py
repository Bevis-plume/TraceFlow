"""
src/utils/config_composer.py
============================
YAML composition utilities for the TraceFlow command-line interface.

The public training scripts still consume one fully resolved YAML file.  This
module builds that file from small, readable fragments: method + data + model +
runtime plus per-run or per-suite overrides.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import yaml

REQUIRED_TOP_LEVEL = ("project", "data", "autoencoder", "model", "training", "sampling", "security", "watermark")
_SECRET_KEYS = {"secret", "secret_key", "private_key"}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep merge where *override* wins."""
    result: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def redact_secrets(obj: Any) -> Any:
    """Return a copy of *obj* with secret-looking fields redacted."""
    if isinstance(obj, Mapping):
        redacted: Dict[str, Any] = {}
        for key, value in obj.items():
            if str(key).lower() in _SECRET_KEYS:
                redacted[key] = "REDACTED"
            else:
                redacted[key] = redact_secrets(value)
        return redacted
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return deepcopy(obj)


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _normalise_override_path(path: str) -> str:
    # Convenience aliases for CLI --set escape hatches.
    aliases = {
        "runtime.training.": "training.",
        "runtime.": "training.",
        "dataset.": "data.",
    }
    for prefix, replacement in aliases.items():
        if path.startswith(prefix):
            return replacement + path[len(prefix):]
    return path


def apply_dotted_override(config: MutableMapping[str, Any], expression: str) -> None:
    """Apply ``a.b.c=value`` to *config* in place."""
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got: {expression}")
    raw_key, raw_value = expression.split("=", 1)
    key = _normalise_override_path(raw_key.strip())
    if not key:
        raise ValueError(f"Override key is empty: {expression}")
    parts = [p for p in key.split(".") if p]
    cursor: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = _parse_scalar(raw_value.strip())


def apply_overrides(config: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    result = deepcopy(config)
    for expr in overrides or []:
        apply_dotted_override(result, expr)
    return result


def validate_resolved_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in config]
    if missing:
        raise ValueError(f"Resolved config missing required sections: {missing}")
    lt = config.get("security", {}).get("latent_transform", {})
    if lt.get("type") == "keyed" and not lt.get("secret_key"):
        raise ValueError("Keyed latent transform requires security.latent_transform.secret_key")
    wm = config.get("watermark", {})
    if wm.get("enabled") and wm.get("type") != "traceflow":
        raise ValueError("Only watermark.type='traceflow' is supported in active configs")


def _load_fragment(path: str | Path, repo_root: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    return load_yaml(p)


def compose_fragments(paths: Iterable[str | Path], repo_root: str | Path = ".") -> Dict[str, Any]:
    repo_root = Path(repo_root)
    merged: Dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, _load_fragment(path, repo_root))
    return merged


def _apply_secret(config: Dict[str, Any], secret_key: Optional[str]) -> None:
    if not secret_key:
        return
    lt = config.setdefault("security", {}).setdefault("latent_transform", {})
    if lt.get("type") == "keyed":
        lt["secret_key"] = secret_key


def _apply_vae_path(config: Dict[str, Any], vae_path: Optional[str]) -> None:
    if vae_path is not None:
        config.setdefault("autoencoder", {})["pretrained_model_name_or_path"] = vae_path


def _apply_run_metadata(config: Dict[str, Any], run: Mapping[str, Any]) -> None:
    if run.get("name"):
        config.setdefault("project", {})["name"] = run["name"]
    if run.get("output_dir"):
        config.setdefault("project", {})["output_dir"] = run["output_dir"]
    if run.get("checkpoint_dir"):
        config.setdefault("training", {})["checkpoint_dir"] = run["checkpoint_dir"]


def compose_run_config(
    run_config_path: str | Path,
    *,
    repo_root: str | Path = ".",
    set_overrides: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compose a single-run YAML into a train_flow_transformer-compatible config.

    Returns ``(resolved_config, run_meta)`` where run_meta contains the original
    run/postprocess blocks and the paths used for composition.
    """
    repo_root = Path(repo_root)
    raw = load_yaml(repo_root / run_config_path if not Path(run_config_path).is_absolute() else run_config_path)
    compose = raw.get("compose", {})
    for key in ("method", "data", "model", "runtime"):
        if key not in compose:
            raise ValueError(f"Run config missing compose.{key}: {run_config_path}")
    resolved = compose_fragments(
        [compose["method"], compose["data"], compose["model"], compose["runtime"]],
        repo_root=repo_root,
    )
    _apply_run_metadata(resolved, raw.get("run", {}))
    _apply_secret(resolved, raw.get("security", {}).get("secret_key"))
    _apply_vae_path(resolved, raw.get("autoencoder", {}).get("pretrained_model_name_or_path"))
    resolved = apply_overrides(resolved, set_overrides)
    validate_resolved_config(resolved)
    return resolved, {
        "run": raw.get("run", {}),
        "compose": compose,
        "postprocess": raw.get("postprocess", {}),
        "source": str(run_config_path),
    }


def compose_suite_experiment(
    suite: Mapping[str, Any],
    exp_id: str,
    *,
    repo_root: str | Path = ".",
    set_overrides: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compose one experiment from a suite YAML."""
    repo_root = Path(repo_root)
    common = suite.get("common", {})
    experiments = suite.get("experiments", {})
    exp = experiments.get(exp_id)
    if exp is None:
        raise ValueError(f"Suite does not define {exp_id}")
    for key in ("method",):
        if key not in exp:
            raise ValueError(f"Suite experiment {exp_id} missing {key}")
    for key in ("data", "model", "runtime"):
        if key not in common:
            raise ValueError(f"Suite common missing {key}")
    resolved = compose_fragments(
        [exp["method"], common["data"], common["model"], common["runtime"]],
        repo_root=repo_root,
    )
    run_name = exp.get("run_name") or f"{exp_id}_{suite.get('suite', {}).get('mode', 'full')}"
    _apply_run_metadata(
        resolved,
        {
            "name": run_name,
            "output_dir": common.get("output_root", "outputs/flow_transformer"),
            "checkpoint_dir": common.get("checkpoint_root", "checkpoints/flow_transformer"),
        },
    )
    _apply_secret(resolved, common.get("secret_key"))
    _apply_vae_path(resolved, common.get("vae_path"))
    if exp.get("overrides"):
        resolved = deep_merge(resolved, exp["overrides"])
    resolved = apply_overrides(resolved, set_overrides)
    validate_resolved_config(resolved)
    return resolved, {
        "exp_id": exp_id,
        "run_name": run_name,
        "attack": exp.get("attack", "latent"),
        "attack_steps": exp.get("attack_steps"),
        "attacker": exp.get("attacker", "no_key"),
    }
