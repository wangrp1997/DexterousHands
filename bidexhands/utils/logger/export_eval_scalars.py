#!/usr/bin/env python3
"""Export eval TensorBoard scalars to CSV (separate from tools.py train curves).

Eval events live under ``<run_dir>/<env>/<algo>/logs_seed*/eval/events.out.tfevents*``.
Typical scalar tags: average_episode_rewards, max_episode_rewards, success_rate,
success_episodes (see algorithms/marl/runner.py log_env).

Examples::

    python bidexhands/utils/logger/export_eval_scalars.py \\
        --root-dir ./bidexhands/logs/shadow_hand_bottle_cap/happo

    python bidexhands/utils/logger/export_eval_scalars.py \\
        --root-dir ./bidexhands/logs/shadow_hand_bottle_cap --recursive \\
        --merged-csv ./bidexhands/logs/shadow_hand_bottle_cap/eval_all.csv

    # 2x2 eval comparison (happo vs mappo), one PNG, read directly from eval tfevents
    python bidexhands/utils/logger/export_eval_scalars.py \\
        --root-dir ./bidexhands/logs/shadow_hand_bottle_cap \\
        --plot-eval-comparison ./bidexhands/logs/shadow_hand_bottle_cap/eval_happo_mappo.png \\
        --plot-only
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def find_eval_event_files(root_dir: str, recursive: bool) -> List[str]:
    """Paths to tfevents files inside ``.../eval/``."""
    root_dir = os.path.abspath(root_dir)
    out: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            if os.path.basename(dirpath) != "eval":
                continue
            for f in filenames:
                if "tfevents" in f:
                    out.append(os.path.join(dirpath, f))
        return sorted(out)

    if not os.path.isdir(root_dir):
        return []
    for name in sorted(os.listdir(root_dir)):
        if not name.startswith("logs_seed"):
            continue
        eval_dir = os.path.join(root_dir, name, "eval")
        if not os.path.isdir(eval_dir):
            continue
        for f in os.listdir(eval_dir):
            if "tfevents" in f:
                out.append(os.path.join(eval_dir, f))
    return sorted(out)


def _parse_algo_seed(eval_event_path: str) -> Tuple[str, str]:
    """Infer algorithm folder and seed name from .../algo/logs_seed-X/eval/file."""
    eval_dir = os.path.dirname(eval_event_path)
    seed_dir = os.path.dirname(eval_dir)
    algo_dir = os.path.dirname(seed_dir)
    seed = os.path.basename(seed_dir)
    algo = os.path.basename(algo_dir)
    return algo, seed


def export_one_tfevent(event_path: str, output_csv: Optional[str] = None) -> str:
    """Write one wide CSV (env_step + all scalar tags). Returns output path."""
    acc = event_accumulator.EventAccumulator(event_path)
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    if not tags:
        raise SystemExit(f"No scalar tags in {event_path!r}")

    # Union of steps: assume same steps across tags (runner logs together)
    by_step: Dict[int, Dict[str, float]] = {}
    for tag in tags:
        for ev in acc.Scalars(tag):
            row = by_step.setdefault(int(ev.step), {})
            row[tag] = float(ev.value)

    steps = sorted(by_step.keys())
    fieldnames = ["env_step"] + tags

    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(event_path), "eval_scalars.csv")

    with open(output_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in steps:
            row: Dict[str, object] = {"env_step": s}
            for t in tags:
                if t in by_step[s]:
                    row[t] = by_step[s][t]
            w.writerow(row)
    return output_csv


def export_merged(event_paths: List[str], output_csv: str) -> None:
    """Single CSV with algo, seed, env_step, and all scalar columns (wide per row)."""
    all_tags: List[str] = []
    rows_out: List[Dict] = []

    for ep in sorted(event_paths):
        acc = event_accumulator.EventAccumulator(ep)
        acc.Reload()
        tags = sorted(acc.Tags().get("scalars", []))
        for t in tags:
            if t not in all_tags:
                all_tags.append(t)

        algo, seed = _parse_algo_seed(ep)
        by_step: Dict[int, Dict[str, float]] = {}
        for tag in tags:
            for ev in acc.Scalars(tag):
                row = by_step.setdefault(int(ev.step), {})
                row[tag] = float(ev.value)
        for s in sorted(by_step.keys()):
            r = {"algorithm": algo, "seed": seed, "env_step": s, "source_event": ep}
            r.update(by_step[s])
            rows_out.append(r)

    fieldnames = ["algorithm", "seed", "env_step"] + all_tags + ["source_event"]
    with open(output_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)


EVAL_METRICS: List[Tuple[str, str]] = [
    ("average_episode_rewards", "Avg episode reward (eval)"),
    ("max_episode_rewards", "Max episode reward (eval)"),
    ("success_rate", "Success rate"),
    ("success_episodes", "Success episodes (cumulative)"),
]


def _scalar_curves_from_tfevent(event_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per tag: (steps, values), duplicate steps collapsed to last value."""
    acc = event_accumulator.EventAccumulator(event_path)
    acc.Reload()
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in acc.Tags().get("scalars", []):
        evs = acc.Scalars(tag)
        if not evs:
            continue
        last: Dict[int, float] = {}
        for ev in evs:
            last[int(ev.step)] = float(ev.value)
        steps = sorted(last.keys())
        out[tag] = (
            np.array(steps, dtype=np.int64),
            np.array([last[s] for s in steps], dtype=np.float64),
        )
    return out


def _pick_eval_tfevent(task_root: str, algo: str, eval_seed: Optional[str]) -> str:
    algo_dir = os.path.join(task_root, algo)
    if not os.path.isdir(algo_dir):
        raise FileNotFoundError(f"Missing algorithm dir: {algo_dir}")
    seeds = sorted([d for d in os.listdir(algo_dir) if d.startswith("logs_seed")])
    if not seeds:
        raise SystemExit(f"No logs_seed* under {algo_dir}")

    candidates: List[str] = []
    if eval_seed:
        if eval_seed not in seeds:
            raise SystemExit(f"--eval-seed {eval_seed!r} not in {seeds} for {algo}")
        candidates = [eval_seed]
    else:
        if "logs_seed-1" in seeds:
            candidates.append("logs_seed-1")
        candidates += [s for s in seeds if s not in candidates]

    for sd in candidates:
        eval_dir = os.path.join(algo_dir, sd, "eval")
        if not os.path.isdir(eval_dir):
            continue
        tfs = [f for f in os.listdir(eval_dir) if "tfevents" in f]
        if tfs:
            return os.path.join(eval_dir, sorted(tfs)[-1])
    raise SystemExit(f"No eval tfevents under {algo_dir} (tried {candidates})")


def plot_eval_comparison_figure(
    task_root: str,
    output_png: str,
    algos: Optional[List[str]] = None,
    eval_seed: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> None:
    """One figure, 2x2 subplots: four eval scalars, all algorithms overlaid."""
    task_root = os.path.abspath(task_root)
    if algos is None:
        algos = ["happo", "mappo"]

    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for algo in algos:
        ep = _pick_eval_tfevent(task_root, algo, eval_seed)
        curves[algo] = _scalar_curves_from_tfevent(ep)
        print(f"[plot] {algo}: {ep}")

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    flat_ax = axes.flatten()
    palette = plt.cm.tab10(np.linspace(0, 0.45, max(len(algos), 2)))

    for ax, (key, title) in zip(flat_ax, EVAL_METRICS):
        drew = False
        for i, algo in enumerate(algos):
            if key not in curves.get(algo, {}):
                continue
            xs, ys = curves[algo][key]
            ax.plot(
                xs,
                ys,
                label=algo,
                color=palette[i % len(palette)],
                linewidth=1.6,
                marker="o",
                markersize=2.5,
            )
            drew = True
        if not drew:
            ax.text(0.5, 0.5, f"No `{key}`", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("TensorBoard step")
        if drew:
            ax.legend(loc="best")
        if key == "success_rate":
            ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    env_name = os.path.basename(task_root)
    fig.suptitle(f"Eval comparison — {env_name} ({', '.join(algos)})", fontsize=13)
    out_abs = os.path.abspath(output_png)
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_abs, dpi=dpi)
    plt.close(fig)
    print("Wrote", out_abs)


def main() -> None:
    p = argparse.ArgumentParser(description="Export eval TB scalars to CSV")
    p.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Either an algorithm dir (.../task/algo) with logs_seed*/eval, or with "
        "--recursive the task root to scan all **/eval/*tfevents*.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Walk entire root-dir for any .../eval/*tfevents* (e.g. task root with "
        "multiple algorithms).",
    )
    p.add_argument(
        "--merged-csv",
        type=str,
        default="",
        help="If set, write one CSV with algorithm, seed, env_step and all tags.",
    )
    p.add_argument(
        "--merged-only",
        action="store_true",
        help="Only write --merged-csv; do not write per-run eval_scalars.csv.",
    )
    p.add_argument(
        "--plot-eval-comparison",
        type=str,
        default="",
        help="Task root (--root-dir): read happo & mappo eval tfevents, save one 2x2 PNG path.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only run --plot-eval-comparison; skip CSV export (useful when --root-dir is task).",
    )
    p.add_argument(
        "--eval-algos",
        type=str,
        default="happo,mappo",
        help="Comma-separated algorithm folders under task root for --plot-eval-comparison.",
    )
    p.add_argument(
        "--eval-seed",
        type=str,
        default="",
        help="Force logs_seed name (e.g. logs_seed-1). Default: prefer logs_seed-1 if present.",
    )
    args = p.parse_args()
    if args.merged_only and not args.merged_csv:
        p.error("--merged-only requires --merged-csv")

    plotted = False
    if args.plot_eval_comparison:
        algos = [a.strip() for a in args.eval_algos.split(",") if a.strip()]
        plot_eval_comparison_figure(
            args.root_dir,
            args.plot_eval_comparison,
            algos=algos,
            eval_seed=args.eval_seed or None,
        )
        plotted = True

    if args.plot_only:
        return

    files = find_eval_event_files(args.root_dir, args.recursive)
    if not files:
        if plotted:
            return
        raise SystemExit(
            f"No eval tfevents under {args.root_dir!r}. "
            "Point --root-dir at .../<algo> (e.g. happo), or use --recursive from task root."
        )

    if args.merged_csv:
        out_m = os.path.abspath(args.merged_csv)
        d = os.path.dirname(out_m)
        if d:
            os.makedirs(d, exist_ok=True)
        export_merged(files, out_m)
        print("Wrote", out_m)

    if not args.merged_only:
        for fpath in files:
            out = export_one_tfevent(fpath, None)
            print("Wrote", out)


if __name__ == "__main__":
    main()
