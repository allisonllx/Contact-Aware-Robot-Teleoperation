
import csv
import math
import statistics
from pathlib import Path

from .io import mean_or_blank, read_json, select_log_path_from_dir
from .schemas import (
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_JAMMING_THRESHOLD_N,
    EXPERIMENT_RESULTS_DIR,
    FILTERED_LOG_NAME,
    FORCE_COMPARISON_CANDIDATES,
    FORCE_ESTIMATION_SCENARIOS,
    FORCE_EST_BY_SCENARIO_COLUMNS,
    FORCE_EST_ERROR_KEYS,
    FORCE_EST_PER_RUN_COLUMNS,
    RAW_LOG_NAME,
    SCENARIOS,
    TRIAL_METADATA_NAME,
)
from .summaries import write_rows
from .telemetry import analyze_result_dir


def write_force_estimation_report(
    root,
    source="auto",
    force_threshold=DEFAULT_FORCE_THRESHOLD_N,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    include_anomalies=False,
    include_tester_pool=True,
    experiment_results_dir=EXPERIMENT_RESULTS_DIR,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    discovered = discover_force_estimation_runs(root)
    if include_tester_pool:
        discovered.extend(discover_tester_pool_runs(Path(experiment_results_dir)))

    per_run_rows = []
    for item in discovered:
        summary = analyze_result_dir(
            result_dir=item["run_dir"],
            scenario=item["scenario"],
            source=source,
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
            include_anomalies=include_anomalies,
        )
        per_run_rows.append(
            {
                "scenario": item["scenario"],
                "run_id": item["run_id"],
                "source": item["source"],
                "run_dir": str(item["run_dir"]),
                "status": summary.get("status", ""),
                "source_csv": summary.get("source_csv", ""),
                "force_comparison_png": find_force_comparison_png(item["run_dir"]),
                "samples_contact_clean": summary.get("samples_contact_clean", ""),
                "mae_contact_n": summary.get("mae_contact_n", ""),
                "mse_contact_n2": summary.get("mse_contact_n2", ""),
                "rmse_contact_n": summary.get("rmse_contact_n", ""),
                "bias_contact_n": summary.get("bias_contact_n", ""),
                "median_abs_error_contact_n": summary.get(
                    "median_abs_error_contact_n", ""
                ),
                "p95_abs_error_contact_n": summary.get("p95_abs_error_contact_n", ""),
                "max_abs_error_contact_n": summary.get("max_abs_error_contact_n", ""),
            }
        )

    by_scenario_rows = aggregate_force_estimation_rows(per_run_rows)

    per_run_path = root / "force_estimation_per_run.csv"
    by_scenario_path = root / "force_estimation_by_scenario.csv"
    write_rows(per_run_path, per_run_rows, FORCE_EST_PER_RUN_COLUMNS)
    write_rows(by_scenario_path, by_scenario_rows, FORCE_EST_BY_SCENARIO_COLUMNS)

    plots_dir = root / "plots"
    plot_force_estimation_bars(by_scenario_rows, per_run_rows, plots_dir)
    plot_force_calibration(
        discovered,
        source=source,
        include_anomalies=include_anomalies,
        plots_dir=plots_dir,
    )
    write_force_estimation_exemplars(per_run_rows, plots_dir / "exemplar_overlays.txt")

    print_force_estimation_report(
        per_run_rows,
        by_scenario_rows,
        per_run_path,
        by_scenario_path,
        plots_dir,
    )

def discover_force_estimation_runs(root):
    """Discover scripted repeats under force_estimation_runs/<scenario>/run_XX/."""
    root = Path(root)
    discovered = []
    if not root.exists():
        return discovered

    scenario_names = list(FORCE_ESTIMATION_SCENARIOS)
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in scenario_names and child.name != "plots":
            if child.name in SCENARIOS:
                scenario_names.append(child.name)

    for scenario in scenario_names:
        scenario_dir = root / scenario
        if not scenario_dir.is_dir():
            continue
        run_dirs = sorted(
            path
            for path in scenario_dir.iterdir()
            if path.is_dir() and path.name.startswith("run_")
        )
        if not run_dirs and select_log_path_from_dir(scenario_dir, "auto") is not None:
            # Allow a single flat folder as run_01 for convenience.
            run_dirs = [scenario_dir]
        for run_dir in run_dirs:
            run_id = run_dir.name if run_dir != scenario_dir else "run_01"
            discovered.append(
                {
                    "scenario": scenario,
                    "run_id": run_id,
                    "source": "scripted",
                    "run_dir": run_dir.resolve(),
                }
            )
    return discovered

def discover_tester_pool_runs(experiment_results_dir):
    """Discover occluded peg_in_hole trial logs under experiment_results/."""
    experiment_results_dir = Path(experiment_results_dir)
    discovered = []
    if not experiment_results_dir.exists():
        return discovered

    seen = set()
    for pattern in (FILTERED_LOG_NAME, RAW_LOG_NAME):
        for log_path in sorted(experiment_results_dir.rglob(pattern)):
            run_dir = log_path.parent.resolve()
            if run_dir in seen:
                continue
            if select_log_path_from_dir(run_dir, "auto") is None:
                continue
            metadata = read_json(run_dir / TRIAL_METADATA_NAME)
            if metadata is not None and metadata.get("status") != "completed":
                continue
            seen.add(run_dir)
            try:
                rel = run_dir.relative_to(experiment_results_dir.resolve())
                run_id = str(rel).replace("\\", "/")
            except ValueError:
                run_id = run_dir.name
                rel = Path(run_id)
            if any(part.casefold() in {"archive", "_archive"} for part in rel.parts):
                continue
            discovered.append(
                {
                    "scenario": "peg_in_hole",
                    "run_id": run_id,
                    "source": "tester",
                    "run_dir": run_dir,
                }
            )
    return discovered

def find_force_comparison_png(run_dir):
    run_dir = Path(run_dir)
    for name in FORCE_COMPARISON_CANDIDATES:
        path = run_dir / name
        if path.exists():
            return str(path.resolve())
    return ""

def aggregate_force_estimation_rows(per_run_rows):
    groups = {}
    for row in per_run_rows:
        key = (row["scenario"], row["source"])
        groups.setdefault(key, []).append(row)

    aggregated = []
    for (scenario, source), rows in sorted(groups.items()):
        ok_rows = [
            row
            for row in rows
            if row.get("status") not in ("missing_csv", "")
            and is_finite_number(row.get("mae_contact_n"))
        ]
        aggregate = {
            "scenario": scenario,
            "source": source,
            "n_runs": len(rows),
            "n_ok": len(ok_rows),
        }
        for key in FORCE_EST_ERROR_KEYS:
            values = [float(row[key]) for row in ok_rows if is_finite_number(row.get(key))]
            aggregate[f"mean_{key}"] = mean_or_blank(values)
            aggregate[f"std_{key}"] = std_or_blank(values)
        aggregated.append(aggregate)
    return aggregated

def plot_force_estimation_bars(by_scenario_rows, per_run_rows, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping force-estimation plots")
        return

    labels = []
    mae_means = []
    mae_stds = []
    mse_means = []
    mse_stds = []
    for row in by_scenario_rows:
        if row["n_ok"] == 0:
            continue
        labels.append(f"{row['scenario']}\n({row['source']})")
        mae_means.append(float(row["mean_mae_contact_n"]))
        mae_stds.append(
            float(row["std_mae_contact_n"])
            if is_finite_number(row["std_mae_contact_n"])
            else 0.0
        )
        mse_means.append(float(row["mean_mse_contact_n2"]))
        mse_stds.append(
            float(row["std_mse_contact_n2"])
            if is_finite_number(row["std_mse_contact_n2"])
            else 0.0
        )

    if labels:
        _save_error_bar_chart(
            plt,
            labels,
            mae_means,
            mae_stds,
            ylabel="MAE (N)",
            title="Contact-force MAE by scenario",
            path=plots_dir / "mae_by_scenario.png",
        )
        _save_error_bar_chart(
            plt,
            labels,
            mse_means,
            mse_stds,
            ylabel="MSE (N²)",
            title="Contact-force MSE by scenario",
            path=plots_dir / "mse_by_scenario.png",
        )

    _save_error_box_plot(
        plt,
        per_run_rows,
        metric_key="mae_contact_n",
        ylabel="MAE (N)",
        title="Contact-force MAE distribution",
        path=plots_dir / "mae_box_by_scenario.png",
    )
    _save_error_box_plot(
        plt,
        per_run_rows,
        metric_key="mse_contact_n2",
        ylabel="MSE (N²)",
        title="Contact-force MSE distribution",
        path=plots_dir / "mse_box_by_scenario.png",
    )


def plot_force_calibration(discovered, *, source, include_anomalies, plots_dir):
    """Plot estimated versus ground-truth contact force, including 0--150 N."""
    plots_dir = Path(plots_dir)
    groups = collect_contact_force_pairs(
        discovered,
        source=source,
        include_anomalies=include_anomalies,
    )
    if not groups:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping force-calibration plots")
        return

    _save_force_calibration_plot(
        plt,
        groups,
        path=plots_dir / "estimate_vs_ground_truth.png",
        title="Contact-force estimate calibration",
    )
    _save_force_calibration_plot(
        plt,
        groups,
        path=plots_dir / "estimate_vs_ground_truth_0_150n.png",
        title="Contact-force estimate calibration (ground truth 0–150 N)",
        limit=150.0,
    )


def collect_contact_force_pairs(discovered, *, source, include_anomalies):
    """Return finite, clean contact (ground truth, estimate) samples by run group."""
    groups = {}
    for item in discovered:
        log_path = select_log_path_from_dir(Path(item["run_dir"]), source)
        if log_path is None:
            continue
        try:
            with log_path.open(newline="") as f:
                rows = csv.DictReader(f)
                for row in rows:
                    true_force = _finite_float(row.get("Ground Truth (N)"))
                    estimate = _finite_float(row.get("Jacobian Estimate (N)"))
                    if true_force is None or estimate is None:
                        continue
                    if not _row_is_contact(row, true_force):
                        continue
                    if not include_anomalies and _row_is_anomaly(row):
                        continue
                    key = (item["scenario"], item["source"])
                    groups.setdefault(key, []).append((true_force, estimate))
        except OSError:
            continue
    return groups


def _save_force_calibration_plot(plt, groups, *, path, title, limit=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    max_value = 0.0
    for (scenario, source), pairs in sorted(groups.items()):
        if limit is not None:
            pairs = [pair for pair in pairs if pair[0] <= limit]
        if not pairs:
            continue
        ground_truth, estimates = zip(*pairs)
        label = f"{scenario} ({source}), n={len(pairs)}"
        ax.scatter(ground_truth, estimates, s=8, alpha=0.18, label=label)
        slope, intercept = _linear_fit(ground_truth, estimates)
        line_end = limit or max(max(ground_truth), max(estimates))
        ax.plot(
            (0.0, line_end),
            (intercept, slope * line_end + intercept),
            linewidth=2,
            label=f"fit: est = {slope:.2f} × GT + {intercept:.1f}",
        )
        max_value = max(max_value, max(ground_truth), max(estimates))
    if max_value == 0.0:
        plt.close(fig)
        return
    x_axis_limit = limit or max_value * 1.05
    y_axis_limit = max(limit or 0.0, max_value * 1.05)
    ax.plot((0.0, x_axis_limit), (0.0, x_axis_limit), "--", color="#333333", label="perfect agreement")
    ax.set_xlim(0.0, x_axis_limit)
    ax.set_ylim(0.0, y_axis_limit)
    if limit is None:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth contact force (N)")
    ax.set_ylabel("Jacobian estimate (N)")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _linear_fit(x_values, y_values):
    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)
    variance_x = sum((value - mean_x) ** 2 for value in x_values)
    if variance_x == 0.0:
        return 0.0, mean_y
    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    )
    slope = covariance / variance_x
    return slope, mean_y - slope * mean_x


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_is_contact(row, true_force):
    if "In Contact" not in row:
        return true_force > 0.0
    return _finite_float(row.get("In Contact")) not in (None, 0.0)


def _row_is_anomaly(row):
    return _finite_float(row.get("Is Anomaly")) not in (None, 0.0)

def _save_error_bar_chart(plt, labels, means, stds, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8", ecolor="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _save_error_box_plot(plt, per_run_rows, metric_key, ylabel, title, path):
    groups = {}
    for row in per_run_rows:
        if not is_finite_number(row.get(metric_key)):
            continue
        if row.get("status") in ("missing_csv", ""):
            continue
        label = f"{row['scenario']}\n({row['source']})"
        groups.setdefault(label, []).append(float(row[metric_key]))
    if not groups:
        return

    labels = sorted(groups)
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.5))
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def write_force_estimation_exemplars(per_run_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representative per-run GT vs estimate overlays",
        "# Prefer contact-only filtered force_comparison_*.png from each group.",
        "",
    ]
    best_by_group = {}
    for row in per_run_rows:
        if not row.get("force_comparison_png"):
            continue
        if not is_finite_number(row.get("mae_contact_n")):
            continue
        key = (row["scenario"], row["source"])
        current = best_by_group.get(key)
        if current is None or float(row["mae_contact_n"]) < float(current["mae_contact_n"]):
            best_by_group[key] = row

    for (scenario, source), row in sorted(best_by_group.items()):
        lines.append(
            f"{scenario} / {source} / {row['run_id']}: {row['force_comparison_png']}"
        )
    if len(lines) == 3:
        lines.append("(no force_comparison PNGs found in discovered runs)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def print_force_estimation_report(
    per_run_rows,
    by_scenario_rows,
    per_run_path,
    by_scenario_path,
    plots_dir,
):
    print(f"Saved per-run force-estimation table to {per_run_path.resolve()}")
    print(f"Saved by-scenario force-estimation table to {by_scenario_path.resolve()}")
    print(f"Plots directory: {plots_dir.resolve()}")
    print()
    print(
        "scenario          source     n_ok/n   MAE mean±std (N)      MSE mean±std (N²)"
    )
    print(
        "----------------- ---------- -------- --------------------- ---------------------"
    )
    for row in by_scenario_rows:
        n_text = f"{row['n_ok']}/{row['n_runs']}"
        mae_text = format_mean_std(row["mean_mae_contact_n"], row["std_mae_contact_n"])
        mse_text = format_mean_std(
            row["mean_mse_contact_n2"], row["std_mse_contact_n2"]
        )
        print(
            f"{str(row['scenario'])[:17]:17} "
            f"{str(row['source'])[:10]:10} "
            f"{n_text:8} "
            f"{mae_text:21} "
            f"{mse_text:21}"
        )
    if not by_scenario_rows:
        print("(no runs found)")
        print(
            "Collect repeats with ./scripts/run_force_estimation_repeats.sh "
            "or copy logs into force_estimation_runs/<scenario>/run_XX/"
        )
    print()
    print(f"Per-run rows analyzed: {len(per_run_rows)}")

def format_mean_std(mean_value, std_value):
    if not is_finite_number(mean_value):
        return ""
    mean_text = f"{float(mean_value):.3g}"
    if not is_finite_number(std_value):
        return mean_text
    return f"{mean_text}±{float(std_value):.3g}"

def is_finite_number(value):
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def std_or_blank(values):
    if len(values) < 2:
        return ""
    return statistics.stdev(values)
