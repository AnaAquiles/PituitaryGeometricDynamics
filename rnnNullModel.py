

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import orthogonal_procrustes, svd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

"""

Low-rank recurrent neural network (RNN) model for validating the geometric
synchrony pipeline under three directed-coupling conditions:

    - Directed  (A → B): g_AB = 3.0, g_BA = 0.0
    - Null      (symmetric): g_AB = 0.0, g_BA = 0.0
    - Reversed  (B → A): g_AB = 0.0, g_BA = 3.0

For each condition the script:
  1. Simulates a rank-2 driver/follower RNN with two populations (PopA, PopB).
  2. Applies the geometric synchrony pipeline:
       - PCA per population (shared component count).
       - Sign alignment of PCA axes (preserves temporal structure for lag).
       - Cross-correlation in raw PCA space to estimate the peak temporal lag.
       - Orthogonal Procrustes alignment of PopB onto PopA (geometry only).
       - Pointwise geometric distance d(t).
       - Sliding-window subspace alignment score S(t).
       - Global principal subspace angles.
  3. Estimates a quasi-potential energy landscape U = -log P via KDE on the
     first two latent dimensions.
  4. Produces two multi-panel validation figures and saves them to disk.

Lag convention
--------------
cross_corr(A, B) peaks at lag t means B[s] ≈ A[s - t]:
  negative lag → A leads B  (B is a delayed copy of A)
  positive lag → B leads A
"""

# ── Simulation parameters ────────────────────────────────────────────────────
N_A        = 50       # units in population A
N_B        = 80       # units in population B
RANK       = 2        # connectivity matrix rank
SR         = 1.15     # spectral radius
T_SIM      = 1200     # total simulation time steps (after burn-in)
DT         = 0.5      # integration time step
TAU        = 5.0      # membrane time constant
NOISE      = 0.015    # additive Gaussian noise amplitude
SEED_W     = 8        # RNG seed for weight matrices
SEED_IC    = 108      # RNG seed for initial conditions

# ── Pipeline parameters ──────────────────────────────────────────────────────
N_COMPONENTS  = 10    # maximum PCA components retained
WINDOW_SIZE   = 20    # half-width of the sliding synchrony window (samples)
LANDSCAPE_RES = 80    # KDE grid resolution for energy landscape
LANDSCAPE_BW  = 0.35  # KDE bandwidth for energy landscape
SMOOTH_SIGMA  = 2.0   # Gaussian smoothing applied to U = -log P
WINDOW_LAND   = 30    # half-width for rolling landscape metrics (samples)

# ── Figure output ────────────────────────────────────────────────────────────
OUT_FIG1 = "fig1_energy_landscape.png"
OUT_FIG2 = "fig2_temporal_shift.png"

COLORS = {
    "directed": "#4e9af1",
    "null":     "#aaaaaa",
    "reversed": "#f47c7c",
    "dim":      ["#ffe082", "#80cbc4", "#ce93d8"],
    "bg":       "#0e0e14",
    "edge":     "#333340",
}


def simulate_rnn(
    n_a: int = N_A,
    n_b: int = N_B,
    rank: int = RANK,
    g_ab: float = 0.0,
    g_ba: float = 0.0,
    sr: float = SR,
    t_sim: int = T_SIM,
    dt: float = DT,
    tau: float = TAU,
    noise: float = NOISE,
    seed_w: int = SEED_W,
    seed_ic: int = SEED_IC,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a rank-``rank`` driver/follower RNN with two populations.

    Each population obeys a continuous-time leaky-integrator update:

        τ dx/dt = -x + W r(x) + g C r(x_other) + η,

    where r(·) = tanh(·), W is a rank-``rank`` recurrent weight matrix scaled
    to spectral radius ``sr``, C is a rank-1 cross-population coupling matrix,
    g is the coupling gain, and η ~ N(0, noise² dt) is additive noise.

    Parameters
    ----------
    n_a, n_b : int
        Number of units in populations A and B.
    rank : int
        Rank of the recurrent connectivity matrices W_A and W_B.
    g_ab : float
        Coupling gain from A to B (A drives B when positive).
    g_ba : float
        Coupling gain from B to A (B drives A when positive).
    sr : float
        Target spectral radius for both recurrent matrices.
    t_sim : int
        Number of time steps to simulate.
    dt, tau : float
        Integration step and membrane time constant; dt/tau controls
        the effective per-step update magnitude.
    noise : float
        Amplitude of additive Gaussian noise.
    seed_w : int
        RNG seed for weight matrix generation (shared across populations
        to ensure comparable geometry).
    seed_ic : int
        RNG seed for initial conditions of population A; population B
        uses seed_ic + 50.

    Returns
    -------
    traj_a : np.ndarray, shape (t_sim, n_a)
        State trajectory of population A.
    traj_b : np.ndarray, shape (t_sim, n_b)
        State trajectory of population B.
    """
    rng_w = np.random.default_rng(seed_w)
    rng_a = np.random.default_rng(seed_ic)
    rng_b = np.random.default_rng(seed_ic + 50)

    def _low_rank_w(n: int, r: int) -> np.ndarray:
        u = rng_w.standard_normal((n, r))
        v = rng_w.standard_normal((r, n))
        w = (u @ v) / np.sqrt(n)
        ev = np.max(np.abs(np.linalg.eigvals(w)))
        return w * (sr / ev) if ev > 1e-10 else w

    w_a  = _low_rank_w(n_a, rank)
    w_b  = _low_rank_w(n_b, rank)
    c_ab = rng_w.standard_normal((n_b, n_a)) / n_a   # A → B
    c_ba = rng_w.standard_normal((n_a, n_b)) / n_b   # B → A

    x_a = rng_a.standard_normal(n_a) * 0.3
    x_b = rng_b.standard_normal(n_b) * 0.3
    traj_a = np.zeros((t_sim, n_a))
    traj_b = np.zeros((t_sim, n_b))
    dt_tau = dt / tau

    for t in range(t_sim):
        r_a, r_b = np.tanh(x_a), np.tanh(x_b)
        x_a += ((-x_a + w_a @ r_a + g_ba * (c_ba @ r_b)) * dt_tau
                + noise * np.sqrt(dt) * rng_a.standard_normal(n_a))
        x_b += ((-x_b + w_b @ r_b + g_ab * (c_ab @ r_a)) * dt_tau
                + noise * np.sqrt(dt) * rng_b.standard_normal(n_b))
        traj_a[t] = x_a
        traj_b[t] = x_b

    return traj_a, traj_b


def geometric_synchrony_pipeline(
    pop_a: np.ndarray,
    pop_b: np.ndarray,
    n_components: int = N_COMPONENTS,
    window: int = WINDOW_SIZE,
) -> dict:
    """Full geometric synchrony analysis pipeline for two population trajectories.

    The pipeline enforces a strict separation between temporal lag estimation
    (which uses sign-aligned raw PCA to preserve temporal structure) and
    geometric comparison (which uses Procrustes-aligned PCA).  Chaining
    Procrustes before cross-correlation would erase the lag by rotating B to
    maximally overlap A across all time points simultaneously.

    Parameters
    ----------
    pop_a : np.ndarray, shape (T, n_a)
        State trajectory of population A.
    pop_b : np.ndarray, shape (T, n_b)
        State trajectory of population B.
    n_components : int
        Maximum number of PCA components to retain per population.
    window : int
        Half-width of the sliding synchrony window in samples.

    Returns
    -------
    dict with keys:
        ZA            : np.ndarray (T, k) — raw PCA trajectory of A.
        ZB_aligned    : np.ndarray (T, k) — Procrustes-aligned PCA of B.
        ZB_sign       : np.ndarray (T, k) — sign-aligned PCA of B (for lag).
        d_t           : np.ndarray (T,)   — pointwise geometric distance.
        sync_score    : np.ndarray        — sliding-window alignment scores.
        sync_times    : np.ndarray        — time indices for sync_score.
        peak_lag      : int               — estimated temporal lag (samples).
        lag_profiles  : list of (lags, corr) tuples — per-dimension profiles.
        subspace_angles : np.ndarray      — global principal angles (degrees).
        mean_distance : float
        mean_sync     : float
        T             : int
    """
    t_len = min(len(pop_a), len(pop_b))
    pop_a, pop_b = pop_a[:t_len], pop_b[:t_len]

    n_comp = min(n_components, pop_a.shape[1], pop_b.shape[1])
    za_raw = PCA(n_components=n_comp).fit_transform(pop_a)
    zb_raw = PCA(n_components=n_comp).fit_transform(pop_b)
    k      = min(za_raw.shape[1], zb_raw.shape[1])
    za_raw, zb_raw = za_raw[:, :k], zb_raw[:, :k]

    # Step 1: sign alignment — minimal correction, preserves temporal structure
    sign_corr = np.sign(np.sum(za_raw * zb_raw, axis=0))
    sign_corr[sign_corr == 0] = 1
    zb_sign = zb_raw * sign_corr

    # Step 2: cross-correlation lag on raw PCA (NOT Procrustes aligned)
    max_lag = t_len // 4
    best_lag, best_val = 0, -np.inf
    lag_profiles = []

    for d in range(min(3, k)):
        a = za_raw[:, d] - za_raw[:, d].mean()
        for sign in [1, -1]:
            b    = sign * (zb_raw[:, d] - zb_raw[:, d].mean())
            corr = np.correlate(a, b, mode="full")
            lags = np.arange(-(t_len - 1), t_len)
            mask = np.abs(lags) <= max_lag
            idx  = np.argmax(corr[mask])
            if corr[mask][idx] > best_val:
                best_val = corr[mask][idx]
                best_lag = lags[mask][idx]

        b0    = zb_sign[:, d] - zb_sign[:, d].mean()
        c     = np.correlate(a, b0, mode="full")
        lags2 = np.arange(-(t_len - 1), t_len)
        m2    = np.abs(lags2) <= max_lag
        norm  = np.abs(c[m2]).max() + 1e-10
        lag_profiles.append((lags2[m2], c[m2] / norm))

    # Step 3: Procrustes alignment — geometry only
    r_mat, _    = orthogonal_procrustes(zb_raw, za_raw)
    zb_aligned  = zb_raw @ r_mat
    d_t         = np.linalg.norm(za_raw - zb_aligned, axis=1)

    # Step 4: sliding-window subspace alignment
    sync_score, sync_times = [], []
    for t in range(window, t_len - window):
        qa, _ = np.linalg.qr(za_raw[t - window : t + window])
        qb, _ = np.linalg.qr(zb_aligned[t - window : t + window])
        _, sigma, _ = svd(qa.T @ qb, full_matrices=False)
        sync_score.append(np.mean(np.clip(sigma, -1.0, 1.0)))
        sync_times.append(t)

    # Step 5: global subspace angles
    qa_g, _ = np.linalg.qr(za_raw)
    qb_g, _ = np.linalg.qr(zb_aligned)
    _, sg, _ = svd(qa_g.T @ qb_g, full_matrices=False)
    angles  = np.degrees(np.arccos(np.clip(sg, -1.0, 1.0)))

    return dict(
        ZA=za_raw, ZB_aligned=zb_aligned, ZB_sign=zb_sign,
        d_t=d_t,
        sync_score=np.array(sync_score),
        sync_times=np.array(sync_times),
        peak_lag=best_lag,
        lag_profiles=lag_profiles,
        subspace_angles=angles,
        mean_distance=float(d_t.mean()),
        mean_sync=float(np.array(sync_score).mean()),
        T=t_len,
    )


def compute_energy_landscape(
    z: np.ndarray,
    grid_res: int = LANDSCAPE_RES,
    bandwidth: float = LANDSCAPE_BW,
    smooth_sigma: float = SMOOTH_SIGMA,
    margin: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a quasi-potential energy landscape via KDE on the first two latent dims.

    The quasi-potential is defined as U(x, y) = -log P(x, y), where P is the
    probability density estimated by a Gaussian KDE.  U is smoothed and
    shifted so that its minimum is zero.

    Parameters
    ----------
    z : np.ndarray, shape (T, k), k >= 2
        Latent trajectory; only the first two dimensions are used.
    grid_res : int
        Number of grid points per axis.
    bandwidth : float
        KDE bandwidth (Scott's method factor).
    smooth_sigma : float
        Standard deviation of the Gaussian smoothing kernel applied to U.
    margin : float
        Extra margin added around the data range on each side.

    Returns
    -------
    Xi, Yi : np.ndarray, shape (grid_res, grid_res)
        Meshgrid coordinates.
    U : np.ndarray, shape (grid_res, grid_res)
        Quasi-potential landscape, U_min = 0.
    """
    x, y = z[:, 0], z[:, 1]
    xi   = np.linspace(x.min() - margin, x.max() + margin, grid_res)
    yi   = np.linspace(y.min() - margin, y.max() + margin, grid_res)
    xi_g, yi_g = np.meshgrid(xi, yi)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    p   = kde(np.vstack([xi_g.ravel(), yi_g.ravel()])).reshape(grid_res, grid_res)
    p   = np.clip(p, 1e-10, None)
    u   = -np.log(p)
    u   = gaussian_filter(u, sigma=smooth_sigma)
    u  -= u.min()
    return xi_g, yi_g, u


def _style_dark_ax(ax, spine_color: str = COLORS["edge"]) -> None:
    """Apply dark-theme styling to a 2-D Matplotlib axes."""
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors="gray")
    for sp in ax.spines.values():
        sp.set_edgecolor(spine_color)


def plot_validation_figure(
    condition_meta: list[tuple],
    out_path: str | None = None,
) -> None:
    """Three-condition validation figure: phase space, time series, cross-correlation.

    Parameters
    ----------
    condition_meta : list of (results_dict, color, title, expected_lag_str)
        One entry per coupling condition.
    out_path : str or None
        If provided, save the figure to this path.
    """
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(COLORS["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, (res, color, title, expected) in enumerate(condition_meta):
        lag      = res["peak_lag"]
        lead_str = "A leads" if lag < -3 else ("B leads" if lag > 3 else "no lead")

        # Phase space
        ax = fig.add_subplot(gs[row, 0])
        _style_dark_ax(ax)
        ax.plot(res["ZA"][:, 0], res["ZA"][:, 1],
                lw=0.8, color=COLORS["directed"], alpha=0.5, label="Pop A")
        ax.plot(res["ZB_aligned"][:, 0], res["ZB_aligned"][:, 1],
                lw=0.8, color=COLORS["reversed"], alpha=0.5, label="Pop B (aligned)")
        ax.scatter(*res["ZA"][0, :2],  color="lime",  s=60, zorder=5, marker="*")
        ax.scatter(*res["ZA"][-1, :2], color="white", s=60, zorder=5, marker="X")
        ax.set(xlabel="Dim 1", ylabel="Dim 2", title=f"{title} — phase space")
        ax.legend(fontsize=7, facecolor="#1a1a24", labelcolor="white")
        ax.grid(True, alpha=0.2)

        # Time series (sign-aligned, not Procrustes)
        ax = fig.add_subplot(gs[row, 1])
        _style_dark_ax(ax)
        t_ax = np.arange(len(res["ZA"]))
        ax.plot(t_ax, res["ZA"][:, 0],      lw=1.0, color=COLORS["directed"],
                alpha=0.85, label="Pop A")
        ax.plot(t_ax, res["ZB_sign"][:, 0], lw=1.0, color=COLORS["reversed"],
                alpha=0.85, label="Pop B (sign-aligned)")
        ax.set(xlabel="Time (steps)", ylabel="Dim-1 projection",
               title=f"Time series — lag = {lag:+d} ({lead_str})")
        ax.legend(fontsize=7, facecolor="#1a1a24", labelcolor="white")
        ax.grid(True, alpha=0.2)

        # Cross-correlation profiles
        ax = fig.add_subplot(gs[row, 2])
        _style_dark_ax(ax)
        for d, (lags_w, corr_w) in enumerate(res["lag_profiles"]):
            ax.plot(lags_w, corr_w, lw=1.2, color=COLORS["dim"][d],
                    alpha=0.8, label=f"Dim {d + 1}")
        ax.axvline(lag, ls="--", color="white", lw=1.8,
                   label=f"Peak lag = {lag:+d}\n(expected: {expected})")
        ax.axvline(0,   ls=":",  color="gray",  lw=0.8)
        ax.set(xlabel="Lag (steps)", ylabel="Norm. cross-correlation",
               xlim=(-80, 80),
               title="Cross-correlation (raw PCA, no Procrustes)")
        ax.legend(fontsize=6, facecolor="#1a1a24", labelcolor="white")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Low-rank RNN validation — three coupling conditions\n"
        "Lag convention: negative = A leads B  |  positive = B leads A",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {out_path}")
    plt.show()


def plot_energy_landscape_figure(
    res: dict,
    snapshot_indices: list[tuple[int, int]],
    snapshot_labels: list[str],
    snapshot_colors: list[str],
    out_path: str | None = None,
) -> None:
    """Temporal energy landscape figure: 3-D surface + 2-D contour per snapshot.

    Parameters
    ----------
    res : dict
        Pipeline output for a single condition.
    snapshot_indices : list of (start, end)
        Time-index pairs defining each temporal window.
    snapshot_labels, snapshot_colors : list of str
        Labels and colours for each snapshot.
    out_path : str or None
        If provided, save the figure here.
    """
    za, zb = res["ZA"], res["ZB_aligned"]
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(COLORS["bg"])

    for idx, (start, end) in enumerate(snapshot_indices):
        z_window = np.vstack([za[start:end, :2], zb[start:end, :2]])
        xi, yi, u = compute_energy_landscape(z_window, grid_res=60, bandwidth=0.35)

        # 3-D surface
        ax3 = fig.add_subplot(2, 4, idx + 1, projection="3d")
        ax3.set_facecolor(COLORS["bg"])
        ax3.plot_surface(xi, yi, u, cmap="plasma", alpha=0.72,
                         linewidth=0, antialiased=True)
        z_floor = u.min() - 0.15
        for z_traj, col in [(za[start:end], "#7eb8f7"),
                             (zb[start:end], "#f47c7c")]:
            ax3.plot(z_traj[:, 0], z_traj[:, 1],
                     np.full(len(z_traj), z_floor), color=col, lw=1.4, alpha=0.9)
            ax3.scatter(z_traj[0, 0],  z_traj[0, 1],  z_floor,
                        color="lime",  s=35, zorder=5)
            ax3.scatter(z_traj[-1, 0], z_traj[-1, 1], z_floor,
                        color="white", s=35, marker="X", zorder=5)
        ax3.set_title(snapshot_labels[idx], color="white", fontsize=9, pad=4)
        ax3.set_xlabel("Dim 1", color="gray", fontsize=7, labelpad=1)
        ax3.set_ylabel("Dim 2", color="gray", fontsize=7, labelpad=1)
        ax3.set_zlabel("U = −log P", color="gray", fontsize=7, labelpad=1)
        ax3.tick_params(colors="gray", labelsize=5)
        for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(COLORS["edge"])
        ax3.view_init(elev=28, azim=-55)

        # 2-D contour
        ax2 = fig.add_subplot(2, 4, idx + 5)
        _style_dark_ax(ax2)
        xi2, yi2, u2 = compute_energy_landscape(z_window, grid_res=80, bandwidth=0.35)
        ax2.contourf(xi2, yi2, u2, levels=20, cmap="plasma", alpha=0.85)
        ax2.contour( xi2, yi2, u2, levels=10, colors="white",
                     alpha=0.15, linewidths=0.5)
        for z_traj, cmap_name in [(za[start:end], "Blues"),
                                   (zb[start:end], "Reds")]:
            seg   = len(z_traj)
            cgrad = plt.get_cmap(cmap_name)(np.linspace(0.35, 1.0, seg))
            for i in range(seg - 1):
                ax2.plot(z_traj[i:i+2, 0], z_traj[i:i+2, 1],
                         color=cgrad[i], lw=1.0, alpha=0.85)
            ax2.scatter(z_traj[0, 0],  z_traj[0, 1],  color="lime",  s=28, zorder=5)
            ax2.scatter(z_traj[-1, 0], z_traj[-1, 1], color="white", s=28,
                        marker="X", zorder=5)
        ax2.set_title(f"Top view — {snapshot_labels[idx]}", color="white", fontsize=8)
        ax2.set_xlabel("Dim 1", color="gray", fontsize=7)
        ax2.set_ylabel("Dim 2", color="gray", fontsize=7)

    fig.text(0.5, 0.51,
             "● Start  ✕ End  |  Blue = Pop A  |  Red = Pop B (Procrustes aligned)",
             ha="center", color="gray", fontsize=8)
    fig.suptitle(
        "Temporal energy landscape — population geometric synchrony\n"
        "Surface: U = −log P(dim₁, dim₂)  |  each column = equal temporal window",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"Saved: {out_path}")
    plt.show()


def plot_temporal_shift_figure(
    res: dict,
    snapshot_indices: list[tuple[int, int]],
    snapshot_colors: list[str],
    window_land: int = WINDOW_LAND,
    out_path: str | None = None,
) -> None:
    """Rolling landscape metrics figure: d(t), well depth, centroid separation.

    Parameters
    ----------
    res : dict
        Pipeline output for a single condition.
    snapshot_indices : list of (start, end)
        Temporal windows to shade across all panels.
    snapshot_colors : list of str
        Colours for the shaded snapshot bands.
    window_land : int
        Half-width for the rolling landscape window (samples).
    out_path : str or None
        If provided, save the figure here.
    """
    za, zb = res["ZA"], res["ZB_aligned"]
    d_t    = res["d_t"]
    t_len  = res["T"]

    well_depth, centroid_sep, times_w = [], [], []
    for t in range(window_land, t_len - window_land):
        z_w = np.vstack([za[t - window_land : t + window_land, :2],
                         zb[t - window_land : t + window_land, :2]])
        _, _, u_w = compute_energy_landscape(
            z_w, grid_res=40, bandwidth=0.4, smooth_sigma=1.5
        )
        well_depth.append(u_w.max() - u_w.min())
        c_a = za[t - window_land : t + window_land, :2].mean(axis=0)
        c_b = zb[t - window_land : t + window_land, :2].mean(axis=0)
        centroid_sep.append(np.linalg.norm(c_a - c_b))
        times_w.append(t)

    well_depth    = np.array(well_depth)
    centroid_sep  = np.array(centroid_sep)
    times_w       = np.array(times_w)
    peak_dom      = times_w[np.argmax(centroid_sep)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in axes:
        _style_dark_ax(ax)

    # Panel A — d(t)
    axes[0].plot(np.arange(t_len), d_t, color="#b084f5", lw=1.2, alpha=0.9)
    axes[0].fill_between(np.arange(t_len), d_t, alpha=0.18, color="#b084f5")
    axes[0].axhline(d_t.mean(), ls="--", color="gray", lw=0.9,
                    label=f"Mean = {d_t.mean():.3f}")
    axes[0].set_ylabel(r"$\Delta(t)$ = $\|Z_A - Z_B\|$", color="white", fontsize=9)
    axes[0].set_title("A — geometric distance between populations",
                      color="white", fontsize=10)
    axes[0].legend(fontsize=8, facecolor="#1a1a24", labelcolor="white")

    # Panel B — well depth
    axes[1].plot(times_w, well_depth, color="#f4a261", lw=1.5)
    axes[1].fill_between(times_w, well_depth, alpha=0.18, color="#f4a261")
    axes[1].axhline(well_depth.mean(), ls="--", color="gray", lw=0.9,
                    label=f"Mean = {well_depth.mean():.3f}")
    axes[1].set_ylabel("Well depth  ($U_{\\max} - U_{\\min}$)", color="white", fontsize=9)
    axes[1].set_title("B — energy landscape depth: attractor strength over time",
                      color="white", fontsize=10)
    axes[1].legend(fontsize=8, facecolor="#1a1a24", labelcolor="white")

    # Panel C — centroid separation
    axes[2].plot(times_w, centroid_sep, color="#4fc3f7", lw=1.5)
    axes[2].fill_between(times_w, centroid_sep, alpha=0.18, color="#4fc3f7")
    axes[2].axhline(centroid_sep.mean(), ls="--", color="gray", lw=0.9,
                    label=f"Mean = {centroid_sep.mean():.3f}")
    axes[2].axvline(peak_dom, color="white", ls=":", lw=1.3,
                    label=f"Peak dominance t = {peak_dom}")
    axes[2].set_ylabel("Centroid separation", color="white", fontsize=9)
    axes[2].set_xlabel("Time (steps)", color="white", fontsize=9)
    axes[2].set_title("C — population centroid separation: geometric dominance",
                      color="white", fontsize=10)
    axes[2].legend(fontsize=8, facecolor="#1a1a24", labelcolor="white", ncol=2)

    # Shade snapshot windows
    for i, (start, end) in enumerate(snapshot_indices):
        for ax in axes:
            ax.axvspan(start, end, alpha=0.07, color=snapshot_colors[i])
            ylim = ax.get_ylim()
            ax.text((start + end) / 2, ylim[0] + (ylim[1] - ylim[0]) * 0.88,
                    f"W{i + 1}", color=snapshot_colors[i], fontsize=8,
                    ha="center", fontweight="bold")

    fig.suptitle(
        "Temporal landscape dynamics — geometric synchrony shift\n"
        "Shaded bands = temporal windows shown in the energy landscape figure",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"Saved: {out_path}")
    plt.show()


def print_validation_summary(condition_meta: list[tuple]) -> None:
    """Print a structured validation summary to stdout."""
    print("=" * 65)
    print("VALIDATION SUMMARY")
    print("Lag convention: negative = A leads B, positive = B leads A")
    print("=" * 65)
    for res, _, label, expected in condition_meta:
        lag  = res["peak_lag"]
        lead = "A leads" if lag < -3 else ("B leads" if lag > 3 else "no lead")
        ok   = (
            (expected.startswith("negative") and lag < -3) or
            (expected.startswith("positive") and lag >  3) or
            (expected.startswith("near")     and abs(lag) <= 15)
        )
        print(f"\n  {'✓' if ok else '✗'} {label}")
        print(f"     Lag               : {lag:+4d}  ({lead})")
        print(f"     Mean sync score   : {res['mean_sync']:.4f}")
        print(f"     Mean distance     : {res['mean_distance']:.4f}")
        print(f"     Mean subsp. angle : {res['subspace_angles'].mean():.1f}°")
    print("=" * 65)


if __name__ == "__main__":
    # ── Simulate three coupling conditions ───────────────────────────────────
    print("Simulating RNNs...")
    pop_a_dir,  pop_b_dir  = simulate_rnn(g_ab=3.0, g_ba=0.0)
    pop_a_null, pop_b_null = simulate_rnn(g_ab=0.0, g_ba=0.0)
    pop_a_rev,  pop_b_rev  = simulate_rnn(g_ab=0.0, g_ba=3.0)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    print("Running geometric synchrony pipeline...")
    res_dir  = geometric_synchrony_pipeline(pop_a_dir,  pop_b_dir)
    res_null = geometric_synchrony_pipeline(pop_a_null, pop_b_null)
    res_rev  = geometric_synchrony_pipeline(pop_a_rev,  pop_b_rev)

    condition_meta = [
        (res_dir,  COLORS["directed"], "Directed (A→B)",  "negative (A leads)"),
        (res_null, COLORS["null"],     "Null (symmetric)", "near zero"),
        (res_rev,  COLORS["reversed"], "Reversed (B→A)",  "positive (B leads)"),
    ]

    print_validation_summary(condition_meta)
    plot_validation_figure(condition_meta)

    # ── Energy landscape figures (directed condition) ─────────────────────────
    t_len = res_dir["T"]
    n_snaps = 4
    snap_size    = t_len // n_snaps
    snap_indices = [(i * snap_size, (i + 1) * snap_size) for i in range(n_snaps)]
    snap_labels  = [f"t = {s}–{e}" for s, e in snap_indices]
    snap_colors  = ["#4e9af1", "#f4a261", "#e76f51", "#2a9d8f"]

    print("Plotting energy landscape figure (Figure 1)...")
    plot_energy_landscape_figure(
        res_dir, snap_indices, snap_labels, snap_colors, out_path=OUT_FIG1
    )

    print("Computing rolling landscape metrics (Figure 2) — takes ~30 s...")
    plot_temporal_shift_figure(
        res_dir, snap_indices, snap_colors, out_path=OUT_FIG2
    )

    print("Done.")
