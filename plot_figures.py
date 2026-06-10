import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# ---- data transcribed from the 750-episode run log (2026-06-10) -----------
# In-run evals: 20 deterministic episodes on the fixed seed list, every 25 eps.
EVAL_EPS = np.arange(0, 751, 25)
EVAL_SUCC = np.array([95, 95, 90, 85, 75, 85, 80, 80, 85, 90, 75, 60, 75, 85,
                      80, 65, 80, 90, 90, 85, 95, 85, 95, 95, 95, 95, 95, 100,
                      85, 100, 90], dtype=float)

# Diagnostics sampled at each eval (ep 25..750).
DIAG_EPS = np.arange(25, 751, 25)
BC_LOSS = np.array([.0035, .0039, .0044, .0049, .0052, .0059, .0064, .0069,
                    .0067, .0070, .0069, .0068, .0069, .0071, .0072, .0074,
                    .0076, .0075, .0078, .0080, .0074, .0079, .0075, .0076,
                    .0078, .0077, .0075, .0080, .0085, .0086])
MEAN_Q = np.array([8.7, 8.6, 8.9, 9.7, 9.3, 8.7, 8.3, 7.5, 6.6, 5.9, 5.3, 4.7,
                   4.3, 3.9, 3.4, 3.3, 2.9, 3.0, 2.8, 2.8, 2.7, 2.8, 2.6, 2.8,
                   2.7, 3.1, 3.1, 3.0, 3.0, 3.1])
BC_LAMBDA = np.concatenate([[10.00, 9.40, 8.78, 8.15, 7.53, 6.90, 6.28, 5.65,
                             5.03], np.full(22, 5.00)])
# Rolling 50-episode stochastic-rollout stats from the progress bar.
TRAIN_SUCC = np.array([4, 4, 2, 0, 0, 0, 2, 4, 4, 6, 4, 2, 8, 16, 18, 20, 22,
                       26, 36, 48, 44, 36, 38, 36, 40, 50, 44, 34, 32, 32],
                      dtype=float)
AVG_R = np.array([0.84, 0.82, 0.59, 0.34, 0.35, 0.34, 0.49, 0.73, 0.69, 0.92,
                  0.66, 0.46, 1.34, 2.41, 2.71, 2.96, 3.21, 3.81, 5.25, 6.82,
                  6.25, 5.15, 5.43, 5.22, 5.71, 7.09, 6.38, 5.06, 4.67, 4.63])

# Committed 40-episode paired eval (paired_eval.log), fallback for Figure 3.
STAGES = ["approach_lid", "lid_open", "approach_sphere", "grasp",
          "transport", "drop"]
PAIRED40 = {"bc": [100, 100, 95, 95, 95, 95],
            "sac": [100, 100, 97.5, 97.5, 97.5, 97.5]}

BC_COLOR, SAC_COLOR = "#888888", "#d95f02"


def wilson(p, n, z=1.96):
    """Wilson score interval half-widths -> (lower_err, upper_err)."""
    p = np.asarray(p, dtype=float)
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p - (center - half), (center + half) - p


def fig1_success_vs_episode(sweep_path="checkpoint_sweep.json"):
    if not os.path.exists(sweep_path):
        print(f"[fig1] {sweep_path} not found - run eval_checkpoints.py first")
        return
    with open(sweep_path) as f:
        data = json.load(f)
    if not data.get("sweep"):
        print("[fig1] sweep has no checkpoints yet - skipping")
        return
    n = data["episodes"]
    eps = [0] + [c["episode"] for c in data["sweep"]]
    succ = [data["bc"]["success_rate"]] + \
           [c["success_rate"] for c in data["sweep"]]
    succ = np.array(succ) * 100
    lo, hi = wilson(succ / 100, n)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bc_p = data["bc"]["success_rate"]
    bc_lo, bc_hi = wilson(np.array([bc_p]), n)
    ax.axhspan((bc_p - bc_lo[0]) * 100, (bc_p + bc_hi[0]) * 100,
               color=BC_COLOR, alpha=0.15, zorder=0)
    ax.axhline(bc_p * 100, color=BC_COLOR, ls="--", lw=1.5,
               label=f"BC baseline ({bc_p:.0%})")
    ax.errorbar(eps, succ, yerr=[lo * 100, hi * 100], fmt="o-",
                color=SAC_COLOR, lw=1.8, ms=4.5, capsize=3,
                label="BC+SAC checkpoint")
    ax.set_xlabel("SAC training episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("SAC fine-tuning (100 episodes per checkpoint in eval)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig1_success_vs_episode.png", dpi=200)
    print("[fig1] wrote figures/fig1_success_vs_episode.png")


def fig2_training_diagnostics():
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.2), sharex=True)

    ax = axes[0]
    lo, hi = wilson(EVAL_SUCC / 100, 20)
    ax.errorbar(EVAL_EPS, EVAL_SUCC, yerr=[lo * 100, hi * 100], fmt="o-",
                color=SAC_COLOR, lw=1.6, ms=3.5, capsize=2.5, alpha=0.9)
    ax.axhline(95, color=BC_COLOR, ls="--", lw=1.3, label="BC init (95%)")
    ax.set_ylabel("Eval success (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Training diagnostics")

    ax = axes[1]
    # drop the ep-25 point: the trailing-50 mean only exists from ep 50 on
    # (before that the logged value averaged all episodes so far)
    ax.plot(DIAG_EPS[1:], AVG_R[1:], "^-", color="#7570b3", lw=1.4, ms=3,
            label="avg episode reward (last 50)")
    ax.set_ylabel("Avg reward")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    ax.plot(DIAG_EPS, BC_LOSS, "o-", color="#e7298a", lw=1.6, ms=3.5,
            label="bc_loss")
    ax.set_ylabel("bc_loss (MSE)")
    ax.set_ylim(0, 0.012)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[3]
    ax.plot(EVAL_EPS, BC_LAMBDA, "-", color="#66a61e", lw=1.8,
            label="lambda_bc")
    ax.axhline(5.0, color="k", ls=":", lw=1.2)
    ax.set_ylabel("lambda_bc")
    ax.set_ylim(0, 11)
    ax.set_xlabel("SAC training episode")
    ax.legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig2_training_diagnostics.png", dpi=200)
    print("[fig2] wrote figures/fig2_training_diagnostics.png")


def fig4_dagger_iterations(sweep_path="dagger_sweep.json"):
    if not os.path.exists(sweep_path):
        print(f"[fig4] {sweep_path} not found - run eval_dagger.py first")
        return
    with open(sweep_path) as f:
        data = json.load(f)
    if not data.get("sweep"):
        print("[fig4] sweep has no policies yet - skipping")
        return
    iters = [c["iteration"] for c in data["sweep"]]
    succ = np.array([c["success_rate"] for c in data["sweep"]]) * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(iters, succ, "o-", color="#1b9e77", lw=1.8, ms=4.5,
            label="BC after DAgger iteration")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("DAgger Iterations")
    ax.set_ylim(0, 102)
    ax.set_xticks(iters)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig4_dagger_iterations.png", dpi=200)
    print("[fig4] wrote figures/fig4_dagger_iterations.png")


def fig3_stage_funnel(sweep_path="checkpoint_sweep.json"):
    n = 40
    bc_rates, sac_rates = PAIRED40["bc"], PAIRED40["sac"]
    src = "40-episode paired eval"
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            data = json.load(f)
        if data.get("sweep"):
            best = max(data["sweep"], key=lambda c: c["success_rate"])
            n = data["episodes"]
            bc_rates = [data["bc"]["stage_completion"][s] * 100 for s in STAGES]
            sac_rates = [best["stage_completion"][s] * 100 for s in STAGES]
            src = f"{n}-episode paired eval, best checkpoint (ep{best['episode']})"

    x = np.arange(len(STAGES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, bc_rates, w, color=BC_COLOR, label="BC")
    ax.bar(x + w / 2, sac_rates, w, color=SAC_COLOR, label="BC+SAC")
    for xi, (b, s) in enumerate(zip(bc_rates, sac_rates)):
        ax.text(xi - w / 2, b + 0.7, f"{b:g}", ha="center", fontsize=8)
        ax.text(xi + w / 2, s + 0.7, f"{s:g}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in STAGES])
    ax.set_ylabel("completion rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Per task completion (100 episodes + best ckpt taken)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("figures/fig3_stage_funnel.png", dpi=200)
    print(f"[fig3] wrote figures/fig3_stage_funnel.png ({src})")


if __name__ == "__main__":
    fig1_success_vs_episode()
    fig2_training_diagnostics()
    fig3_stage_funnel()
    fig4_dagger_iterations()
