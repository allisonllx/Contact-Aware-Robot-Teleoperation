"""Experiment-plan creation, counterbalancing, trial specs, and resume state."""

import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("no_feedback", "visual_feedback", "audio_feedback", "both_feedback")
WILLIAMS_ORDERS = (
    ("no_feedback", "visual_feedback", "both_feedback", "audio_feedback"),
    ("visual_feedback", "audio_feedback", "no_feedback", "both_feedback"),
    ("audio_feedback", "both_feedback", "visual_feedback", "no_feedback"),
    ("both_feedback", "no_feedback", "audio_feedback", "visual_feedback"),
)
TRIAL_METADATA_NAME = "trial_metadata.json"
RAW_LOG_NAME = "force_verification_log.csv"
FILTERED_LOG_NAME = "force_verification_log_filtered.csv"


def sanitize_tester_name(name):
    if not name:
        raise ValueError("Tester name cannot be empty")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-").lower()
    if not cleaned:
        raise ValueError("Tester name must contain at least one letter or number")
    return cleaned


def ensure_selected_conditions_in_plan(selected_conditions, plan):
    assigned = set(plan.get("condition_order", []))
    missing = [condition for condition in selected_conditions if condition not in assigned]
    if missing:
        raise ValueError("Selected conditions are not present in this tester's saved plan: " + ", ".join(missing))


def load_or_create_plan(tester_name, tester_id, tester_dir, experiment_root, selected_conditions, manual_order, familiarization_trials, practice_trials, recorded_trials, max_trial_duration, base_seed):
    plan_path = tester_dir / "experiment_plan.json"
    if plan_path.exists() and manual_order is None:
        with plan_path.open() as f:
            plan = json.load(f)
        if plan.get("base_seed") is None and plan.get("generated_seed_base") is None:
            plan["generated_seed_base"] = secrets.randbits(32)
            plan["updated_at"] = utc_now()
            write_json(plan_path, plan)
        plan["loaded_existing_plan"] = True
        return plan
    if manual_order is not None:
        condition_order, order_index = list(manual_order), "manual"
    else:
        condition_order, order_index = assign_counterbalanced_order(experiment_root, tester_id, selected_conditions)
    now = utc_now()
    plan = {
        "tester": tester_name, "tester_id": tester_id, "scenario": "peg_in_hole",
        "conditions": list(selected_conditions), "condition_order": condition_order,
        "order_index": order_index, "familiarization_trials": familiarization_trials,
        "practice_trials": practice_trials, "recorded_trials": recorded_trials,
        "max_trial_duration": max_trial_duration, "base_seed": base_seed,
        "generated_seed_base": secrets.randbits(32) if base_seed is None else None,
        "created_at": now, "updated_at": now, "loaded_existing_plan": False,
    }
    write_json(plan_path, plan)
    return plan


def assign_counterbalanced_order(experiment_root, tester_id, selected_conditions):
    orders = candidate_orders(selected_conditions)
    counts = [0] * len(orders)
    for plan_path in Path(experiment_root).glob("*/experiment_plan.json") if Path(experiment_root).exists() else ():
        try:
            with plan_path.open() as f:
                existing_order = tuple(json.load(f).get("condition_order", []))
        except (OSError, json.JSONDecodeError):
            continue
        for index, order in enumerate(orders):
            if existing_order == tuple(order):
                counts[index] += 1
    candidates = [index for index, count in enumerate(counts) if count == min(counts)]
    choice = candidates[stable_int(tester_id) % len(candidates)]
    return list(orders[choice]), choice


def candidate_orders(selected_conditions):
    selected_conditions = tuple(selected_conditions)
    if selected_conditions == CONDITIONS:
        return WILLIAMS_ORDERS
    return tuple(selected_conditions[index:] + selected_conditions[:index] for index in range(len(selected_conditions)))


def stable_int(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def build_trial_specs(args, tester_name, tester_id, tester_dir, plan, selected_conditions):
    run_order = [condition for condition in plan["condition_order"] if condition in selected_conditions]
    if not run_order:
        raise ValueError("No selected conditions appear in the assigned experiment order")
    seed_base = plan["base_seed"] if plan.get("base_seed") is not None else plan["generated_seed_base"]
    specs = []
    for trial_index in range(1, args.familiarization_trials + 1):
        trial = _trial_spec(
            tester_name,
            tester_id,
            tester_dir / "familiarization" / f"familiarization_{trial_index:02d}",
            "no_feedback",
            0,
            "familiarization",
            trial_index,
            seed_base,
        )
        trial["seed"] = trial_seed(
            seed_base, tester_id, "familiarization", "familiarization", trial_index
        )
        specs.append(trial)
    for position, condition in enumerate(run_order, start=1):
        for trial_type, count in (("practice", args.practice_trials), ("recorded", args.recorded_trials)):
            for trial_index in range(1, count + 1):
                specs.append(_trial_spec(tester_name, tester_id, tester_dir / condition / f"{trial_type}_{trial_index:02d}", condition, position, trial_type, trial_index, seed_base))
    return specs


def _trial_spec(tester, tester_id, trial_dir, condition, position, trial_type, index, seed_base):
    name = f"{trial_type}_{index:02d}"
    return {"tester": tester, "tester_id": tester_id, "condition": condition, "condition_position": position, "trial_type": trial_type, "trial_index": index, "trial_name": name, "trial_dir": trial_dir, "seed": trial_seed(seed_base, tester_id, condition, trial_type, index), "visual_feedback": condition in ("visual_feedback", "both_feedback"), "audio_feedback": condition in ("audio_feedback", "both_feedback")}


def trial_seed(seed_base, tester_id, condition, trial_type, trial_index):
    return stable_int(f"{seed_base}:{tester_id}:{condition}:{trial_type}:{trial_index}") % (2 ** 32)


def trial_state(trial_dir):
    trial_dir = Path(trial_dir)
    metadata = read_json_if_exists(trial_dir / TRIAL_METADATA_NAME)
    status = metadata.get("status") if isinstance(metadata, dict) else None
    raw, filtered = (trial_dir / RAW_LOG_NAME).exists(), (trial_dir / FILTERED_LOG_NAME).exists()
    return {"status": status or "not_started", "has_metadata": metadata is not None, "has_raw_csv": raw, "has_filtered_csv": filtered, "has_telemetry": raw or filtered, "complete": status == "completed" and (raw or filtered)}


def trial_completed(trial_dir):
    return trial_state(trial_dir)["complete"]


def trial_status_for_display(state):
    if state["complete"]:
        return "status=completed"
    if state["status"] == "not_started" and state["has_telemetry"]:
        return "status=partial/no metadata"
    if state["status"] == "completed":
        return "status=completed/missing telemetry"
    if state["has_telemetry"]:
        return f"status={state['status']}/partial telemetry"
    return f"status={state['status']}"


def read_json_if_exists(path):
    if not path.exists():
        return None
    try:
        with path.open() as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"status": "metadata_unreadable"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable.pop("loaded_existing_plan", None)
    with path.open("w") as file:
        json.dump(serializable, file, indent=2, sort_keys=True)
        file.write("\n")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
