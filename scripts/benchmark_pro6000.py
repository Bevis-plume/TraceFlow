"""
scripts.benchmark_pro6000
=========================
Short real-training stress tests for RTX PRO 6000 96GB.

The benchmark launches the existing TraceFlow training entries with several
micro-batch / grad-accumulation settings, records OOMs, wall time, step time,
and CUDA peak memory, then writes JSON and Markdown reports.  It measures the
actual training path rather than a synthetic matmul benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

STAGES: Dict[str, Dict[str, Any]] = {
    "generator": {
        "entry": "train-generator",
        "default_candidates": [(64, 1), (80, 1), (96, 1), (112, 1), (128, 1)],
    },
    "keyed": {
        "entry": "train-keyed",
        "default_candidates": [(64, 1), (80, 1), (96, 1), (112, 1), (128, 1)],
    },
    "identity": {
        "entry": "train-identity",
        "default_candidates": [(32, 3), (40, 2), (48, 2), (56, 2), (64, 2), (72, 1), (80, 1)],
    },
    "traceflow": {
        "entry": "train-final",
        "default_candidates": [(32, 3), (40, 2), (48, 2), (56, 2), (64, 2), (72, 1), (80, 1)],
    },
}


def _parse_candidates(spec: Optional[str], defaults: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spec:
        return defaults
    out: List[Tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Candidate must look like batchxaccum, got {item!r}")
        b, g = item.split("x", 1)
        out.append((int(b), int(g)))
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _target_steps(config_path: str, explicit: Optional[int]) -> int:
    if explicit:
        return explicit
    cfg = _load_yaml(config_path)
    return int(cfg.get("training", {}).get("num_steps", 200000))


def _entry_run_name(config_path: str, entry: str) -> str:
    cfg = _load_yaml(config_path)
    fallback = entry.replace("train-", "traceflow-bench-").replace("-", "_")
    return str(cfg.get("entries", {}).get(entry.replace("-", "_"), {}).get("run_name") or fallback)


def _find_report(bundle: Path) -> Optional[Path]:
    reports = sorted(bundle.glob("outputs/*/train_report.json"))
    if reports:
        return reports[-1]
    smoke_reports = sorted(bundle.glob("outputs/*/smoke_report.json"))
    if smoke_reports:
        return smoke_reports[-1]
    return None


def _run_trial(
    *,
    config: str,
    root: Path,
    stage: str,
    batch_size: int,
    grad_accum: int,
    steps: int,
    extra_sets: Iterable[str],
) -> Dict[str, Any]:
    entry = STAGES[stage]["entry"]
    trial_name = f"{stage}_bs{batch_size}_ga{grad_accum}"
    trial_bundle = root / "trials" / trial_name
    log_path = root / "logs" / f"{trial_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trial_bundle.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "-B",
        "-m",
        "scripts.traceflow",
        entry,
        "--config",
        config,
        "--bundle-dir",
        str(trial_bundle),
        "--foreground",
        "--set",
        f"training.batch_size={batch_size}",
        "--set",
        f"training.grad_accum_steps={grad_accum}",
        "--set",
        f"training.num_steps={steps}",
        "--set",
        f"training.log_interval={max(10, min(50, steps // 4 or 1))}",
        "--set",
        f"training.sample_interval={steps + 1}",
        "--set",
        f"training.save_interval={steps + 1}",
        "--set",
        "training.clean_fp_interval=0",
    ]
    for override in extra_sets:
        cmd.extend(["--set", override])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n")
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
        rc = proc.wait()
    wall = time.time() - t0
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    oom = "out of memory" in log_text.lower() or "cuda error: out of memory" in log_text.lower()

    result: Dict[str, Any] = {
        "stage": stage,
        "entry": entry,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "benchmark_steps": steps,
        "returncode": rc,
        "status": "oom" if oom else ("ok" if rc == 0 else "failed"),
        "wall_time_s": wall,
        "log_path": str(log_path),
        "bundle_dir": str(trial_bundle),
    }

    report_path = _find_report(trial_bundle)
    if report_path and report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result["train_report_path"] = str(report_path)
            result["avg_step_time_s"] = report.get("avg_step_time_s")
            result["total_train_wall_time_s"] = report.get("total_train_wall_time_s")
            result["cuda_max_memory_allocated_MB"] = report.get("cuda_max_memory_allocated_MB")
        except json.JSONDecodeError:
            result["report_error"] = "invalid_json"
    return result


def _fmt_hours(seconds: Optional[float]) -> str:
    if seconds is None or math.isnan(seconds):
        return "n/a"
    hours = seconds / 3600.0
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


def _select_best(results: List[Dict[str, Any]], min_effective_batch: int = 64) -> Dict[str, Optional[Dict[str, Any]]]:
    ok = [r for r in results if r.get("status") == "ok" and r.get("avg_step_time_s")]
    if not ok:
        return {"fastest_ok": None, "fastest_comparable": None}
    fastest_ok = min(ok, key=lambda r: float(r["avg_step_time_s"]))
    comparable = [r for r in ok if int(r.get("effective_batch_size", 0)) >= min_effective_batch]
    fastest_comparable = min(comparable, key=lambda r: float(r["avg_step_time_s"])) if comparable else None
    return {"fastest_ok": fastest_ok, "fastest_comparable": fastest_comparable}


def _write_reports(root: Path, results: List[Dict[str, Any]], target_steps: int) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_stage.setdefault(str(row["stage"]), []).append(row)

    selections = {stage: _select_best(rows) for stage, rows in by_stage.items()}
    payload = {
        "target_steps": target_steps,
        "results": results,
        "selections": selections,
    }
    (reports_dir / "pro6000_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# RTX PRO 6000 TraceFlow Benchmark",
        "",
        f"Target training steps for estimates: `{target_steps}`",
        "",
        "## Trial Results",
        "",
        "| stage | batch | accum | effective | status | avg step | peak mem | 200k/target estimate | log |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in results:
        avg = r.get("avg_step_time_s")
        estimate = float(avg) * target_steps if avg else None
        peak = r.get("cuda_max_memory_allocated_MB")
        peak_s = "n/a" if peak is None else f"{float(peak) / 1024.0:.1f} GB"
        avg_s = "n/a" if avg is None else f"{float(avg):.3f}s"
        lines.append(
            f"| {r['stage']} | {r['batch_size']} | {r['grad_accum_steps']} | "
            f"{r['effective_batch_size']} | {r['status']} | {avg_s} | {peak_s} | "
            f"{_fmt_hours(estimate)} | `{r['log_path']}` |"
        )

    lines.extend(["", "## Recommended Fastest Comparable Settings", ""])
    total = 0.0
    total_known = True
    for stage in STAGES:
        if stage not in selections:
            continue
        best = selections[stage].get("fastest_comparable") or selections[stage].get("fastest_ok")
        if not best:
            lines.append(f"- `{stage}`: no successful trial")
            total_known = False
            continue
        avg = float(best["avg_step_time_s"])
        total += avg * target_steps
        lines.append(
            f"- `{stage}`: batch `{best['batch_size']}`, accum `{best['grad_accum_steps']}`, "
            f"effective `{best['effective_batch_size']}`, avg `{avg:.3f}s/step`, "
            f"estimate `{_fmt_hours(avg * target_steps)}`"
        )
    if total_known and total > 0:
        lines.extend(["", f"Estimated full four-checkpoint training time: **{_fmt_hours(total)}**"])
    lines.extend([
        "",
        "Notes:",
        "- This benchmark measures the actual TraceFlow training entry, including VAE/data/model overhead.",
        "- Prefer `fastest comparable` over a smaller effective batch if paper settings should stay consistent.",
        "- OOM trials are expected during stress testing and are not code failures.",
    ])
    (reports_dir / "pro6000_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[benchmark] report: {reports_dir / 'pro6000_benchmark.md'}", flush=True)
    print(f"[benchmark] json:   {reports_dir / 'pro6000_benchmark.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TraceFlow batch sizes on RTX PRO 6000.")
    parser.add_argument("--config", default="configs/traceflow.yml")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--stages", default="all", help="all or comma list: generator,keyed,identity,traceflow")
    parser.add_argument("--candidates", default=None, help="Override candidates for all stages, e.g. 32x2,48x1,64x1")
    parser.add_argument("--steps", type=int, default=300, help="Short training steps per trial.")
    parser.add_argument("--target-steps", type=int, default=None, help="Steps used for runtime estimates. Defaults to config training.num_steps.")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--stop-on-first-oom", action="store_true")
    args = parser.parse_args()

    root = Path(args.bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    if args.stages == "all":
        stages = list(STAGES.keys())
    else:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}. Choose from {list(STAGES)}")

    target_steps = _target_steps(args.config, args.target_steps)
    results: List[Dict[str, Any]] = []
    for stage in stages:
        candidates = _parse_candidates(args.candidates, STAGES[stage]["default_candidates"])
        print(f"[benchmark] stage={stage} candidates={candidates}", flush=True)
        for batch_size, grad_accum in candidates:
            print(f"[benchmark] trial stage={stage} batch={batch_size} accum={grad_accum}", flush=True)
            result = _run_trial(
                config=args.config,
                root=root,
                stage=stage,
                batch_size=batch_size,
                grad_accum=grad_accum,
                steps=args.steps,
                extra_sets=args.set_overrides,
            )
            results.append(result)
            _write_reports(root, results, target_steps)
            if args.stop_on_first_oom and result.get("status") == "oom":
                print(f"[benchmark] stopping {stage} after OOM at batch={batch_size}", flush=True)
                break

    _write_reports(root, results, target_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
