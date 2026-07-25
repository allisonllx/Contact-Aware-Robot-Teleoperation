
import math
import statistics

from .io import mean, mean_or_blank, median, percentile, safe_divide, select
from .schemas import EPS


def smoothness_metrics(times, is_clean, in_contact, target_x, target_y, target_z):
    blank = {
        "mean_action_jerk": "",
        "velocity_reversals": "",
        "retraction_count": "",
    }
    points = [
        (t, x, y, z, contact)
        for t, clean, x, y, z, contact in zip(
            times, is_clean, target_x, target_y, target_z, in_contact
        )
        if clean and math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
    ]
    if len(points) < 4:
        return blank

    ts = [p[0] for p in points]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    zs = [p[3] for p in points]
    contacts = [p[4] for p in points]

    vx = finite_differences(xs, ts)
    vy = finite_differences(ys, ts)
    vz = finite_differences(zs, ts)
    ax = finite_differences(vx, ts[1:])
    ay = finite_differences(vy, ts[1:])
    az = finite_differences(vz, ts[1:])
    jx = finite_differences(ax, ts[2:])
    jy = finite_differences(ay, ts[2:])
    jz = finite_differences(az, ts[2:])
    jerks = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(jx, jy, jz)]

    reversals = (
        count_sign_changes(vx)
        + count_sign_changes(vy)
        + count_sign_changes(vz)
    )
    retractions = count_retractions(vz, contacts[1:] if len(contacts) > 1 else contacts)

    return {
        "mean_action_jerk": mean_or_blank(jerks),
        "velocity_reversals": reversals,
        "retraction_count": retractions,
    }

def finite_differences(values, times):
    if len(values) < 2 or len(times) < 2:
        return []
    diffs = []
    count = min(len(values) - 1, len(times) - 1)
    for i in range(count):
        dt = times[i + 1] - times[i]
        if dt <= EPS:
            diffs.append(0.0)
        else:
            diffs.append((values[i + 1] - values[i]) / dt)
    return diffs

def count_sign_changes(values, deadband=1e-4):
    filtered = [0.0 if abs(value) < deadband else value for value in values]
    changes = 0
    prev = 0.0
    for value in filtered:
        if value == 0.0:
            continue
        if prev != 0.0 and value * prev < 0.0:
            changes += 1
        prev = value
    return changes

def count_retractions(vz, contacts, lift_speed=1e-4):
    """Count upward target-move episodes after contact has begun."""
    return _count_retraction_episodes(vz, contacts, lift_speed)

def _count_retraction_episodes(vz, contacts, lift_speed):
    seen_contact = False
    lifting = []
    for i, speed in enumerate(vz):
        contact = contacts[i] if i < len(contacts) else False
        seen_contact = seen_contact or bool(contact)
        lifting.append(seen_contact and speed > lift_speed)
    return count_episodes(lifting)

def count_episodes(mask):
    episodes = 0
    prev = False
    for value in mask:
        current = bool(value)
        if current and not prev:
            episodes += 1
        prev = current
    return episodes

def error_metrics(f_true, f_est, mask, scope):
    if scope == "contact":
        keys = {
            "mae": "mae_contact_n",
            "mse": "mse_contact_n2",
            "rmse": "rmse_contact_n",
            "bias": "bias_contact_n",
            "median_abs": "median_abs_error_contact_n",
            "p95_abs": "p95_abs_error_contact_n",
            "max_abs": "max_abs_error_contact_n",
            "nmae": "nmae_contact_mean_gt",
            "nrmse": "nrmse_contact_mean_gt",
        }
    else:
        keys = {
            "mae": "mae_all_clean_n",
            "mse": "mse_all_clean_n2",
            "rmse": "rmse_all_clean_n",
            "bias": "bias_all_clean_n",
            "p95_abs": "p95_abs_error_all_clean_n",
        }

    selected_true = select(f_true, mask)
    selected_est = select(f_est, mask)
    if not selected_true:
        return {key: "" for key in keys.values()}

    error = [estimate - truth for truth, estimate in zip(selected_true, selected_est)]
    abs_error = [abs(value) for value in error]
    mse = mean([value ** 2 for value in error])
    metrics = {
        keys["mae"]: mean(abs_error),
        keys["mse"]: mse,
        keys["rmse"]: math.sqrt(mse),
        keys["bias"]: mean(error),
        keys["p95_abs"]: percentile(abs_error, 95),
    }

    if scope == "contact":
        mean_gt = mean([abs(value) for value in selected_true])
        metrics.update({
            keys["median_abs"]: median(abs_error),
            keys["max_abs"]: max(abs_error),
            keys["nmae"]: safe_divide(metrics[keys["mae"]], mean_gt),
            keys["nrmse"]: safe_divide(metrics[keys["rmse"]], mean_gt),
        })

    return metrics
