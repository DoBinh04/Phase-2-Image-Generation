import numpy as np
import matplotlib.pyplot as plt
import os

from data import sample_8gaussians, all_centers
from flow_matching import train_flow_matching, euler_sample
from ddpm import train_ddpm, ddim_sample
from metrics import straightness_deviation, path_length_ratio
from nn import MLP

# Works both in Google Colab and in a regular local checkout.
OUT = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

STEP_COUNTS = [1, 5, 10, 50]
N_TRAIN_STEPS = 10000
N_PLOT_SAMPLES = 300
N_TRAJ_SHOW = 40  # how many individual paths to draw per subplot


def plot_trajectories(traj_dict, title, fname, centers):
    fig, axes = plt.subplots(1, len(STEP_COUNTS), figsize=(4.2 * len(STEP_COUNTS), 4.4))
    for ax, n in zip(axes, STEP_COUNTS):
        traj = traj_dict[n]  # (n+1, N, 2)
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", c="black", s=60, zorder=5, label="mode centers")
        for i in range(min(N_TRAJ_SHOW, traj.shape[1])):
            ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.7, alpha=0.6, color="tab:blue")
        ax.scatter(traj[0, :N_TRAJ_SHOW, 0], traj[0, :N_TRAJ_SHOW, 1], c="green", s=10, zorder=4, label="start (noise)")
        ax.scatter(traj[-1, :N_TRAJ_SHOW, 0], traj[-1, :N_TRAJ_SHOW, 1], c="red", s=10, zorder=4, label="end (sample)")
        ax.set_title(f"{n} step{'s' if n > 1 else ''}")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        if n == STEP_COUNTS[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=140)
    plt.close(fig)


def main():
    centers = all_centers()

    print("=== Training Flow Matching VelocityNet ===")
    fm_net, fm_losses = train_flow_matching(n_steps=N_TRAIN_STEPS, batch_size=512, hidden=128, seed=0)
    print("final FM loss:", np.mean(fm_losses[-200:]))

    print("=== Training DDPM baseline (noise prediction) ===")
    ddpm_net, ddpm, ddpm_losses = train_ddpm(n_steps=N_TRAIN_STEPS, batch_size=512, hidden=128, T=1000, seed=0)
    print("final DDPM loss:", np.mean(ddpm_losses[-200:]))

    # -------- Loss curves --------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(fm_losses, lw=0.6)
    ax[0].set_title("Flow Matching training loss")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[1].plot(ddpm_losses, lw=0.6, color="tab:orange")
    ax[1].set_title("DDPM training loss")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(f"{OUT}/00_training_losses.png", dpi=140)
    plt.close(fig)

    # -------- Sample trajectories at multiple step counts --------
    # Use a SHARED set of starting noise points across step counts / methods,
    # so trajectories at 1,5,10,50 steps (and FM vs DDPM) start at the same x0
    # for a fair, directly comparable picture.
    rng = np.random.default_rng(123)
    x_start = rng.normal(size=(N_PLOT_SAMPLES, 2))

    fm_trajs = {}
    fm_straightness = {}
    fm_pathratio = {}
    for n in STEP_COUNTS:
        traj, x0, x1 = euler_sample(fm_net, N_PLOT_SAMPLES, n, x_start=x_start)
        fm_trajs[n] = traj
        fm_straightness[n] = straightness_deviation(traj)
        fm_pathratio[n] = path_length_ratio(traj)

    ddpm_trajs = {}
    ddpm_straightness = {}
    ddpm_pathratio = {}
    for n in STEP_COUNTS:
        traj = ddim_sample(ddpm_net, ddpm, N_PLOT_SAMPLES, n, x_start=x_start)
        ddpm_trajs[n] = traj
        ddpm_straightness[n] = straightness_deviation(traj)
        ddpm_pathratio[n] = path_length_ratio(traj)

    plot_trajectories(fm_trajs, "Flow Matching: Euler sampling trajectories", "01_fm_trajectories.png", centers)
    plot_trajectories(ddpm_trajs, "DDPM baseline: DDIM (deterministic) trajectories", "02_ddpm_trajectories.png", centers)

    # -------- Straightness comparison plot --------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xs = STEP_COUNTS
    ax[0].plot(xs, [fm_straightness[n] for n in xs], "o-", label="Flow Matching")
    ax[0].plot(xs, [ddpm_straightness[n] for n in xs], "s-", label="DDPM (DDIM)")
    ax[0].set_xlabel("Euler/DDIM steps")
    ax[0].set_ylabel("mean normalized deviation from straight chord")
    ax[0].set_title("Trajectory curvature (lower = straighter)")
    ax[0].legend()

    ax[1].plot(xs, [fm_pathratio[n] for n in xs], "o-", label="Flow Matching")
    ax[1].plot(xs, [ddpm_pathratio[n] for n in xs], "s-", label="DDPM (DDIM)")
    ax[1].axhline(1.0, color="gray", ls="--", lw=0.8, label="perfectly straight (=1)")
    ax[1].set_xlabel("Euler/DDIM steps")
    ax[1].set_ylabel("path length / chord length")
    ax[1].set_title("Path-length ratio (1.0 = straight line)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_straightness_comparison.png", dpi=140)
    plt.close(fig)

    print("\nStraightness (normalized deviation), lower=straighter:")
    for n in xs:
        print(f"  steps={n:2d}  FM={fm_straightness[n]:.4f}   DDPM={ddpm_straightness[n]:.4f}")
    print("\nPath length / chord length ratio, 1.0=straight:")
    for n in xs:
        print(f"  steps={n:2d}  FM={fm_pathratio[n]:.4f}   DDPM={ddpm_pathratio[n]:.4f}")

    # -------- 1-step generation quality (qualitative) --------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    data_sample = sample_8gaussians(2000, seed=1)
    for ax, (name, n) in zip(axes, [("Flow Matching", 1), ("DDPM (DDIM)", 1)]):
        if name == "Flow Matching":
            traj, _, x1 = euler_sample(fm_net, 2000, n, seed=7)
        else:
            traj = ddim_sample(ddpm_net, ddpm, 2000, n, seed=7)
            x1 = traj[-1]
        ax.scatter(data_sample[:, 0], data_sample[:, 1], s=4, alpha=0.15, color="gray", label="true data")
        ax.scatter(x1[:, 0], x1[:, 1], s=6, alpha=0.6, color="tab:red", label="1-step samples")
        ax.set_title(f"{name}: 1-step samples")
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_one_step_quality.png", dpi=140)
    plt.close(fig)

    # ==================== BONUS: Reflow ====================
    print("\n=== Reflow bonus ===")
    N_PAIRS = 10000
    REFLOW_GEN_STEPS = 100  # high-accuracy Euler integration to build (x0,x1) pairs
    rng2 = np.random.default_rng(999)
    x0_pairs = rng2.normal(size=(N_PAIRS, 2))
    _, _, x1_pairs = euler_sample(fm_net, N_PAIRS, REFLOW_GEN_STEPS, x_start=x0_pairs)
    print(f"generated {N_PAIRS} deterministic (noise -> sample) pairs using the trained FM model "
          f"({REFLOW_GEN_STEPS}-step Euler integration).")

    reflow_net, reflow_losses = train_flow_matching_on_pairs(
        x0_pairs, x1_pairs, n_steps=N_TRAIN_STEPS, batch_size=512, hidden=128, seed=1
    )
    print("final Reflow loss:", np.mean(reflow_losses[-200:]))

    # Compare curvature before/after reflow, same shared starting noise
    reflow_trajs = {}
    reflow_straightness = {}
    reflow_pathratio = {}
    for n in STEP_COUNTS:
        traj, _, _ = euler_sample(reflow_net, N_PLOT_SAMPLES, n, x_start=x_start)
        reflow_trajs[n] = traj
        reflow_straightness[n] = straightness_deviation(traj)
        reflow_pathratio[n] = path_length_ratio(traj)

    plot_trajectories(reflow_trajs, "After Reflow: Euler sampling trajectories", "05_reflow_trajectories.png", centers)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(xs, [fm_straightness[n] for n in xs], "o-", label="FM (before reflow)")
    ax[0].plot(xs, [reflow_straightness[n] for n in xs], "^-", label="FM (after reflow)")
    ax[0].set_xlabel("Euler steps")
    ax[0].set_ylabel("mean normalized deviation from straight chord")
    ax[0].set_title("Curvature before vs after Reflow")
    ax[0].legend()

    ax[1].plot(xs, [fm_pathratio[n] for n in xs], "o-", label="FM (before reflow)")
    ax[1].plot(xs, [reflow_pathratio[n] for n in xs], "^-", label="FM (after reflow)")
    ax[1].axhline(1.0, color="gray", ls="--", lw=0.8, label="perfectly straight (=1)")
    ax[1].set_xlabel("Euler steps")
    ax[1].set_ylabel("path length / chord length")
    ax[1].set_title("Path-length ratio before vs after Reflow")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_reflow_comparison.png", dpi=140)
    plt.close(fig)

    print("\nPath length / chord ratio before vs after reflow:")
    for n in xs:
        print(f"  steps={n:2d}  before={fm_pathratio[n]:.4f}   after={reflow_pathratio[n]:.4f}")

    # 1-step quality after reflow
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    traj1, _, x1_1step = euler_sample(reflow_net, 2000, 1, seed=7)
    ax.scatter(data_sample[:, 0], data_sample[:, 1], s=4, alpha=0.15, color="gray", label="true data")
    ax.scatter(x1_1step[:, 0], x1_1step[:, 1], s=6, alpha=0.6, color="tab:purple", label="1-step (after reflow)")
    ax.set_title("Reflow: 1-step samples")
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_reflow_one_step_quality.png", dpi=140)
    plt.close(fig)

    print("\nDone. All figures saved to", OUT)


def train_flow_matching_on_pairs(x0_pairs, x1_pairs, n_steps=10000, batch_size=512, lr=1e-3, hidden=128, seed=0):
    """
    Reflow: retrain a fresh VelocityNet using the SAME FM loss, but now the
    (x0, x1) coupling is no longer independent — it's the deterministic pairs
    produced by the previously trained model's ODE. This straightens the
    trajectories further (per the Rectified Flow / Reflow procedure).
    """
    net = MLP([3, hidden, hidden, hidden, 2], seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_pairs = x0_pairs.shape[0]
    losses = []
    for step in range(n_steps):
        idx = rng.integers(0, n_pairs, size=batch_size)
        x0 = x0_pairs[idx]
        x1 = x1_pairs[idx]
        t = rng.uniform(0, 1, size=(batch_size, 1))

        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        inp = np.concatenate([xt, t], axis=1)
        pred_v, cache = net.forward(inp)

        diff = pred_v - target_v
        loss = np.mean(np.sum(diff ** 2, axis=1))
        dout = 2 * diff / batch_size

        dW, db = net.backward(cache, dout)
        net.adam_step(dW, db, lr=lr)
        losses.append(loss)
    return net, losses


if __name__ == "__main__":
    main()
