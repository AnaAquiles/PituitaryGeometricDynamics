"""

Geometric synchrony analysis between two neural population trajectories
(PopA and PopB) in a shared latent space.

The pipeline proceeds in five stages:

  1. Time-delay embedding (Takens' theorem) of each population's
     fluorescence traces into a higher-dimensional state space.
  2. Dimensionality reduction via PCA, yielding per-population latent
     trajectories.
  3. Procrustes alignment of PopB's trajectory into PopA's coordinate
     frame via an optimal orthogonal rotation.
  4. Geometric synchrony quantification:
       - Pointwise trajectory distance d(t).
       - Global principal subspace angles (dominance metric).
       - Sliding-window subspace alignment score (local synchrony).
  5. Temporal KDE to identify epochs of high vs. low synchrony.

Assumed upstream variables
--------------------------
data_g1, data_g2 : dict
    Dictionaries keyed by condition label; values are lists of
    np.ndarray of shape (n_cells, n_samples).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

N_TIMEPOINTS  = 600   # number of time points used per population
N_COMPONENTS  = 10    # maximum PCA components
EMBED_DIM     = 2     # time-delay embedding dimension (m)
EMBED_TAU     = 1     # time-delay embedding lag (tau)
WINDOW_SIZE   = 20    # half-width of the sliding synchrony window (samples)

COLORS = {
    "A":     "royalblue",
    "B":     "tomato",
    "sync":  "mediumseagreen",
    "div":   "darkorange",
    "angle": "mediumpurple",
}


def time_delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Embed a 1-D time series using Takens' time-delay construction.

    Reconstructs a pseudo-phase-space attractor by stacking lagged copies
    of the signal into a matrix of shape (N - (m-1)*tau, m).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Univariate time series.
    m : int
        Embedding dimension.
    tau : int
        Time delay (lag) in samples.

    Returns
    -------
    np.ndarray, shape (N - (m-1)*tau, m)
        Delay-embedded trajectory.
    """
    N = len(x) - (m - 1) * tau
    return np.array([x[i : i + m * tau : tau] for i in range(N)])


def population_embedding(df: pd.DataFrame, m: int, tau: int) -> np.ndarray:
    """Apply time-delay embedding to every cell in a population DataFrame.

    Each cell is embedded independently and the resulting trajectories are
    concatenated horizontally, truncated to the shortest embedding length.

    Parameters
    ----------
    df : pd.DataFrame
        Population fluorescence traces; one column per cell, rows are time.
    m : int
        Embedding dimension.
    tau : int
        Time delay (lag) in samples.

    Returns
    -------
    np.ndarray, shape (T_embedded, n_cells * m)
        Horizontally concatenated embedded trajectories.
    """
    embeddings = [time_delay_embedding(df[col].values, m, tau) for col in df.columns]
    min_len = min(e.shape[0] for e in embeddings)
    return np.hstack([e[:min_len] for e in embeddings])


def subspace_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute principal angles between the column spaces of A and B.

    Principal angles are obtained via QR decomposition of each matrix
    followed by SVD of the cross-product of the orthonormal bases.

    Parameters
    ----------
    A : np.ndarray, shape (T, k)
        First matrix.
    B : np.ndarray, shape (T, k)
        Second matrix.

    Returns
    -------
    np.ndarray, shape (k,)
        Principal angles in degrees, in ascending order.
    """
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    _, sigma, _ = svd(QA.T @ QB, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    return np.degrees(np.arccos(sigma))


def sliding_window_alignment(
    ZA: np.ndarray,
    ZB: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a sliding-window subspace alignment score between two trajectories.

    At each time step $t$, a local segment of length $2 \times$ ``window``
    is extracted from both trajectories.  The mean cosine of the principal
    angles between the two local subspaces is used as an alignment score,
    with values near 1 indicating high geometric synchrony.

    Parameters
    ----------
    ZA : np.ndarray, shape (T, k)
        Latent trajectory of population A.
    ZB : np.ndarray, shape (T, k)
        Latent trajectory of population B, aligned to A's frame.
    window : int
        Half-width of the sliding window in samples.

    Returns
    -------
    times : np.ndarray
        Time indices at which the score is evaluated.
    scores : np.ndarray
        Mean cosine of principal angles at each time step.
    """
    T = ZA.shape[0]
    scores, times = [], []

    for t in range(window, T - window):
        seg_A = ZA[t - window : t + window]
        seg_B = ZB[t - window : t + window]
        QA, _ = np.linalg.qr(seg_A)
        QB, _ = np.linalg.qr(seg_B)
        _, sigma, _ = svd(QA.T @ QB, full_matrices=False)
        scores.append(np.mean(np.clip(sigma, -1.0, 1.0)))
        times.append(t)

    return np.array(times), np.array(scores)


def plot_geometric_synchrony(
    ZA: np.ndarray,
    ZB_aligned: np.ndarray,
    d_t: np.ndarray,
    angles: np.ndarray,
    sync_times: np.ndarray,
    sync_score: np.ndarray,
    window: int,
) -> None:
    """Produce the four-panel geometric synchrony summary figure.

    Panels
    ------
    1. Shared phase space (dim 1 vs dim 2) after Procrustes alignment.
    2. Pointwise trajectory distance d(t) over time.
    3. Sliding-window subspace alignment score.
    4. Principal subspace angles (global dominance).
    5. Temporal KDE of high- vs. low-synchrony epochs.

    Parameters
    ----------
    ZA : np.ndarray, shape (T, k)
        Latent trajectory of population A.
    ZB_aligned : np.ndarray, shape (T, k)
        Procrustes-aligned latent trajectory of population B.
    d_t : np.ndarray, shape (T,)
        Pointwise geometric distance between trajectories.
    angles : np.ndarray, shape (k,)
        Global principal subspace angles in degrees.
    sync_times : np.ndarray
        Time indices of the sliding-window alignment scores.
    sync_score : np.ndarray
        Sliding-window alignment scores.
    window : int
        Half-width used for the sliding window (shown in panel title).
    """
    T = len(d_t)
    time_axis = np.arange(T)
    mean_angle = np.mean(angles)

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: shared phase space
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ZA[:, 0], ZA[:, 1], "o", color=COLORS["A"],
             alpha=0.35, ms=3, label="Pop A")
    ax1.plot(ZB_aligned[:, 0], ZB_aligned[:, 1], "o", color=COLORS["B"],
             alpha=0.35, ms=3, label="Pop B (Procrustes aligned)")
    ax1.scatter(*ZA[0, :2],  c="green", s=250, marker="*", zorder=5, label="Start")
    ax1.scatter(*ZA[-1, :2], c="black", s=250, marker="X", zorder=5, label="End")
    ax1.set(xlabel="Latent dim 1", ylabel="Latent dim 2",
            title="Shared population phase space — Procrustes aligned")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # Panel 2: pointwise distance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, d_t, lw=1.2, color="purple", alpha=0.85)
    ax2.axhline(d_t.mean(), ls="--", color="gray", lw=1,
                label=f"Mean = {d_t.mean():.2f}")
    ax2.fill_between(time_axis, d_t, d_t.mean(),
                     where=(d_t < d_t.mean()),  alpha=0.25, color=COLORS["sync"], label="High sync")
    ax2.fill_between(time_axis, d_t, d_t.mean(),
                     where=(d_t >= d_t.mean()), alpha=0.25, color=COLORS["div"],  label="Low sync")
    ax2.set(xlabel="Time (frames)", ylabel=r"$\Delta(t) = \|Z_A - Z_B\|$",
            title="Pointwise geometric distance $\\Delta(t)$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    # Panel 3: sliding-window alignment
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(sync_times, sync_score, lw=1.5, color=COLORS["sync"])
    ax3.axhline(sync_score.mean(), ls="--", color="gray", lw=1,
                label=f"Mean = {sync_score.mean():.3f}")
    ax3.fill_between(sync_times, sync_score, sync_score.mean(),
                     where=(sync_score > sync_score.mean()),  alpha=0.25,
                     color=COLORS["sync"], label="High alignment")
    ax3.fill_between(sync_times, sync_score, sync_score.mean(),
                     where=(sync_score <= sync_score.mean()), alpha=0.25,
                     color=COLORS["div"],  label="Low alignment")
    ax3.set(xlabel="Time (frames)", ylabel="Mean cos(principal angle)",
            title=f"Sliding-window subspace alignment (window = {window})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)

    # Panel 4: principal angles
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(range(len(angles)), angles, color=COLORS["angle"],
            alpha=0.8, edgecolor="white")
    ax4.axhline(45, ls="--", color="gray", lw=1, label="45° (random)")
    ax4.axhline(mean_angle, ls=":", color="black", lw=1.2,
                label=f"Mean = {mean_angle:.1f}°")
    ax4.set(xlabel="Principal angle index", ylabel="Angle (degrees)",
            title="Principal subspace angles — global geometric dominance\n"
                  "(<45° → shared structure,  >45° → divergent geometry)")
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.4)

    # Panel 5: temporal KDE
    ax5 = fig.add_subplot(gs[2, 1])
    high_thresh = np.percentile(sync_score, 75)
    low_thresh  = np.percentile(sync_score, 25)
    high_times  = sync_times[sync_score >= high_thresh]
    low_times   = sync_times[sync_score <= low_thresh]
    t_range     = np.linspace(0, T, 400)

    for times, color, label in [
        (high_times, COLORS["sync"], "High sync (top 25%)"),
        (low_times,  COLORS["div"],  "Low sync / dominance (bottom 25%)"),
    ]:
        if len(times) > 1:
            kde = gaussian_kde(times, bw_method=0.2)
            ax5.plot(t_range, kde(t_range), lw=2, color=color, label=label)
            ax5.fill_between(t_range, kde(t_range), alpha=0.2, color=color)

    ax5.set(xlabel="Time (frames)", ylabel="Density",
            title="Temporal KDE — synchrony / dominance epochs")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.4)

    fig.suptitle("Geometric synchrony in population dynamics",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Data selection ---
    condition_key = "Condition_2"
    subject_idx   = 3

    raw_A = data_g1[condition_key][subject_idx][:, :N_TIMEPOINTS].T  # (T, cells)
    raw_B = data_g2[condition_key][subject_idx][:, :N_TIMEPOINTS].T

    n_cells_A, n_cells_B = raw_A.shape[1], raw_B.shape[1]
    dfA = pd.DataFrame(raw_A, columns=[f"A_cell_{i}" for i in range(n_cells_A)])
    dfB = pd.DataFrame(raw_B, columns=[f"B_cell_{i}" for i in range(n_cells_B)])

    # --- Time-delay embedding ---
    XA = population_embedding(dfA, m=EMBED_DIM, tau=EMBED_TAU)
    XB = population_embedding(dfB, m=EMBED_DIM, tau=EMBED_TAU)

    T = min(len(XA), len(XB))
    XA, XB = XA[:T], XB[:T]

    # --- PCA per population ---
    n_components = min(N_COMPONENTS, XA.shape[1], XB.shape[1])
    ZA = PCA(n_components=n_components).fit_transform(XA)
    ZB = PCA(n_components=n_components).fit_transform(XB)

    # --- Procrustes alignment of ZB onto ZA ---
    R, _ = orthogonal_procrustes(ZB, ZA)
    ZB_aligned = ZB @ R

    # --- Geometric distance ---
    d_t = np.linalg.norm(ZA - ZB_aligned, axis=1)

    # --- Global principal subspace angles ---
    angles = subspace_angles(ZA, ZB_aligned)
    print(f"Mean principal angle: {np.mean(angles):.2f}°")

    # --- Sliding-window alignment score ---
    sync_times, sync_score = sliding_window_alignment(ZA, ZB_aligned, window=WINDOW_SIZE)

    # --- Visualisation ---
    plot_geometric_synchrony(
        ZA, ZB_aligned, d_t, angles, sync_times, sync_score, window=WINDOW_SIZE
    )
