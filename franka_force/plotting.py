import csv

import numpy as np


def plot_force_comparison(env):
    if not env.time_history:
        print("No force samples recorded; skipping plot.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(env.time_history)
    f_true = np.array(env.true_force_history)
    f_est = np.array(env.estimated_force_history)
    in_contact = np.array(env.in_contact_history, dtype=bool)
    is_anomaly = np.array(env.anomaly_history, dtype=bool)
    is_clean = ~is_anomaly
    cushion_active = _history_array(env, "cushion_active_history", len(t), dtype=bool)
    cushion_scale = _history_array(env, "cushion_scale_history", len(t), dtype=float)
    impedance_tau_norm = _history_array(env, "impedance_tau_norm_history", len(t), dtype=float)
    contact_force_vectors = _contact_vector_history(env, len(t))
    task_success = _history_array(env, "task_success_history", len(t), dtype=bool)
    success_contact = _history_array(env, "success_contact_history", len(t), dtype=bool)
    success_hold_time = _history_array(env, "success_hold_time_history", len(t), dtype=float)
    hole_clearance = _history_array(env, "hole_clearance_history", len(t), dtype=float)
    occluded_hole_randomized = _history_array(env, "occluded_hole_randomized_history", len(t), dtype=bool)
    occluded_hole_x = _history_array(env, "occluded_hole_x_history", len(t), dtype=float)
    occluded_hole_y = _history_array(env, "occluded_hole_y_history", len(t), dtype=float)
    occluded_hole_offset_x = _history_array(env, "occluded_hole_offset_x_history", len(t), dtype=float)
    occluded_hole_offset_y = _history_array(env, "occluded_hole_offset_y_history", len(t), dtype=float)
    occluder_alpha = _history_array(env, "occluder_alpha_history", len(t), dtype=float)
    occluder_style = _history_values(env, "occluder_style_history", len(t), default="")
    audio_feedback = _history_array(env, "audio_feedback_history", len(t), dtype=bool)
    audio_contact_event = _history_array(env, "audio_contact_event_history", len(t), dtype=bool)
    audio_tick_rate = _history_array(env, "audio_tick_rate_history", len(t), dtype=float)
    audio_lateral_force = _history_array(env, "audio_lateral_force_history", len(t), dtype=float)
    target_x = _history_array(env, "target_x_history", len(t), dtype=float)
    target_y = _history_array(env, "target_y_history", len(t), dtype=float)
    target_z = _history_array(env, "target_z_history", len(t), dtype=float)

    _write_filtered_csv(
        env.telemetry_filtered_path,
        t,
        f_true,
        f_est,
        in_contact,
        is_clean,
        cushion_active,
        cushion_scale,
        impedance_tau_norm,
        contact_force_vectors,
        task_success,
        success_contact,
        success_hold_time,
        hole_clearance,
        occluded_hole_randomized,
        occluded_hole_x,
        occluded_hole_y,
        occluded_hole_offset_x,
        occluded_hole_offset_y,
        occluder_alpha,
        occluder_style,
        audio_feedback,
        audio_contact_event,
        audio_tick_rate,
        audio_lateral_force,
        target_x,
        target_y,
        target_z,
    )

    _save_plot(
        plt,
        t,
        f_true,
        f_est,
        title="Raw: Measured vs. Estimated Contact Forces",
        path=env.plot_raw_path,
    )

    if np.any(is_clean):
        _save_plot(
            plt,
            t[is_clean],
            f_true[is_clean],
            f_est[is_clean],
            title="Filtered: Measured vs. Estimated Contact Forces",
            path=env.plot_filtered_path,
        )
    else:
        print("No clean samples left after filtering; skipped filtered full plot.")

    if np.any(in_contact):
        _save_plot(
            plt,
            t[in_contact],
            f_true[in_contact],
            f_est[in_contact],
            title="Raw (Contact-Only): Measured vs. Estimated Contact Forces",
            path=env.plot_contact_raw_path,
        )

        contact_clean = in_contact & is_clean
        if np.any(contact_clean):
            _save_plot(
                plt,
                t[contact_clean],
                f_true[contact_clean],
                f_est[contact_clean],
                title="Filtered (Contact-Only): Measured vs. Estimated Contact Forces",
                path=env.plot_contact_filtered_path,
            )
        else:
            print("No clean contact samples; skipped filtered contact-only plot.")
    else:
        print("No target contacts recorded; skipped contact-only plots.")

    n_anomaly = int(np.sum(is_anomaly))
    print(f"Flagged {n_anomaly}/{len(t)} samples as anomalies "
          f"({100 * n_anomaly / len(t):.1f}%)")


def _history_array(env, name, expected_len, dtype=float):
    values = getattr(env, name, [])
    if len(values) != expected_len:
        return np.zeros(expected_len, dtype=dtype)
    return np.array(values, dtype=dtype)


def _contact_vector_history(env, expected_len):
    values = getattr(env, "contact_force_vector_history", [])
    if len(values) != expected_len:
        return np.zeros((expected_len, 3), dtype=float)
    return np.array(values, dtype=float)


def _history_values(env, name, expected_len, default=""):
    values = getattr(env, name, [])
    if len(values) != expected_len:
        return [default] * expected_len
    return values


def _write_filtered_csv(
    path,
    times,
    true_forces,
    est_forces,
    in_contact,
    is_clean,
    cushion_active,
    cushion_scale,
    impedance_tau_norm,
    contact_force_vectors,
    task_success,
    success_contact,
    success_hold_time,
    hole_clearance,
    occluded_hole_randomized,
    occluded_hole_x,
    occluded_hole_y,
    occluded_hole_offset_x,
    occluded_hole_offset_y,
    occluder_alpha,
    occluder_style,
    audio_feedback,
    audio_contact_event,
    audio_tick_rate,
    audio_lateral_force,
    target_x,
    target_y,
    target_z,
):
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time (s)",
            "Ground Truth (N)",
            "Jacobian Estimate (N)",
            "In Contact",
            "Cushion Active",
            "Cushion Scale",
            "Impedance Tau Norm",
            "Contact Force X (N)",
            "Contact Force Y (N)",
            "Contact Force Z (N)",
            "Task Success",
            "Success Contact",
            "Success Hold Time",
            "Hole Clearance (mm)",
            "Occluded Hole Randomized",
            "Occluded Hole X (m)",
            "Occluded Hole Y (m)",
            "Occluded Hole Offset X (m)",
            "Occluded Hole Offset Y (m)",
            "Occluder Alpha",
            "Occluder Style",
            "Audio Feedback",
            "Audio Contact Event",
            "Audio Tick Rate (Hz)",
            "Audio Lateral Force (N)",
            "Target X (m)",
            "Target Y (m)",
            "Target Z (m)",
        ])
        for i in np.where(is_clean)[0]:
            writer.writerow([
                times[i],
                true_forces[i],
                est_forces[i],
                int(in_contact[i]),
                int(cushion_active[i]),
                cushion_scale[i],
                impedance_tau_norm[i],
                contact_force_vectors[i, 0],
                contact_force_vectors[i, 1],
                contact_force_vectors[i, 2],
                int(task_success[i]),
                int(success_contact[i]),
                success_hold_time[i],
                hole_clearance[i],
                int(occluded_hole_randomized[i]),
                occluded_hole_x[i],
                occluded_hole_y[i],
                occluded_hole_offset_x[i],
                occluded_hole_offset_y[i],
                occluder_alpha[i],
                occluder_style[i],
                int(audio_feedback[i]),
                int(audio_contact_event[i]),
                audio_tick_rate[i],
                audio_lateral_force[i],
                target_x[i],
                target_y[i],
                target_z[i],
            ])
    print(f"Saved filtered CSV to {path.resolve()}")


def _save_plot(plt, times, true_forces, est_forces, title, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, true_forces,
            label="Ground Truth (contact, world frame)", color="black", linewidth=2.5)
    ax.plot(times, est_forces,
            label="Jacobian Estimate (qfrc_constraint)", color="orange", linestyle="--", linewidth=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Simulation Time (Seconds)")
    ax.set_ylabel("Force Amplitude (Newtons)")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved force plot to {path.resolve()}")
