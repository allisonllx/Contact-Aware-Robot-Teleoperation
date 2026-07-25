
import csv
import json
import math
import statistics
from pathlib import Path

from .schemas import (
    DEFAULT_JAMMING_THRESHOLD_N,
    EPS,
    FILTERED_LOG_NAME,
    RAW_LOG_NAME,
    SUMMARY_COLUMNS,
)


def read_json(path):
    try:
        with Path(path).open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def select_log_path(results_dir, scenario, source):
    return select_log_path_from_dir(results_dir / scenario, source)

def select_log_path_from_dir(result_dir, source):
    filtered_path = result_dir / FILTERED_LOG_NAME
    raw_path = result_dir / RAW_LOG_NAME

    if source == "filtered":
        return filtered_path if filtered_path.exists() else None
    if source == "raw":
        return raw_path if raw_path.exists() else None
    if filtered_path.exists():
        return filtered_path
    if raw_path.exists():
        return raw_path
    return None

def empty_summary(
    scenario,
    status,
    force_threshold,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    source_csv="",
    used_filtered_csv=False,
):
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update({
        "scenario": scenario,
        "status": status,
        "source_csv": str(source_csv),
        "used_filtered_csv": int(used_filtered_csv),
        "force_threshold_n": force_threshold,
        "jamming_threshold_n": jamming_threshold,
    })
    return row

def optional_float_column(rows, columns, name, default=0.0):
    if name not in columns:
        return [default] * len(rows)
    return float_column(rows, name, default)

def optional_bool_column(rows, columns, name):
    if name not in columns:
        return [False] * len(rows)
    return bool_column(rows, name)

def optional_text_column(rows, columns, name):
    if name not in columns:
        return [""] * len(rows)
    return [row.get(name, "") for row in rows]

def float_column(rows, name, default=0.0):
    values = []
    for row in rows:
        raw_value = row.get(name, "")
        if raw_value in ("", None):
            values.append(default)
            continue
        try:
            values.append(float(raw_value))
        except ValueError:
            values.append(default)
    return values

def bool_column(rows, name):
    return [bool(value) for value in float_column(rows, name)]

def and_masks(*masks):
    return [all(values) for values in zip(*masks)]

def count_true(mask):
    return sum(1 for value in mask if value)

def select(values, mask):
    return [value for value, selected in zip(values, mask) if selected]

def sample_widths(times):
    if len(times) == 0:
        return []
    if len(times) == 1:
        return [0.0]

    diffs = [max(b - a, 0.0) for a, b in zip(times, times[1:])]
    positive_diffs = [value for value in diffs if value > 0.0]
    final_width = median(positive_diffs) if positive_diffs else 0.0
    return diffs + [final_width]

def duration(times):
    if len(times) < 2:
        return 0.0
    return max(times[-1] - times[0], 0.0)

def first_time(times, mask):
    for time, selected in zip(times, mask):
        if selected:
            return time
    return ""

def weighted_duration(widths, mask):
    return sum(width for width, selected in zip(widths, mask) if selected)

def weighted_sum(values, widths, mask):
    return sum(
        value * width
        for value, width, selected in zip(values, widths, mask)
        if selected
    )

def safe_divide(numerator, denominator):
    denominator = float(denominator)
    if abs(denominator) <= EPS:
        return ""
    return float(numerator) / denominator

def mean_or_blank(values):
    if not values:
        return ""
    return mean(values)

def max_or_blank(values):
    if not values:
        return ""
    return max(values)

def finite_median(values):
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return ""
    return median(finite_values)

def first_nonempty(values):
    for value in values:
        if value:
            return value
    return ""

def mean(values):
    return sum(values) / len(values)

def median(values):
    if not values:
        return ""
    return percentile(values, 50)

def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in SUMMARY_COLUMNS})

def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value

def print_summary(rows, output_path):
    print(f"Saved analysis summary to {output_path.resolve()}")
    print()
    print("scenario        status       contact n   MAE (N)   RMSE (N)   bias (N)   peak GT (N)")
    print("--------------- ------------ ----------- --------- ---------- ---------- ------------")
    for row in rows:
        print(
            f"{str(row['scenario'])[:15]:15} "
            f"{str(row['status'])[:12]:12} "
            f"{display(row['samples_contact_clean'], width=11)} "
            f"{display(row['mae_contact_n'], width=9)} "
            f"{display(row['rmse_contact_n'], width=10)} "
            f"{display(row['bias_contact_n'], width=10)} "
            f"{display(row['peak_ground_truth_contact_n'], width=12)}"
        )

def display(value, width):
    if value == "":
        return " " * width
    if isinstance(value, float):
        text = f"{value:.3g}"
    else:
        text = str(value)
    return text[:width].rjust(width)
