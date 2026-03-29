
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelmax
from scipy.stats import gaussian_kde

"""

Quasi-potential energy landscape analysis with dominance angle estimation
for one-dimensional population signals.

For each group (all data, Class 1, Class 2) and each physiological condition,
the script:
  1. Estimates the quasi-potential U(x) = -log P(x) from the empirical
     signal distribution via a 1-D Gaussian KDE (Silverman bandwidth).
  2. Identifies the two deepest potential wells and the energy barrier
     separating them, characterising bistability in the population dynamics.
  3. Computes the dominance angle θ = arctan(ΔU / Δx), a signed scalar
     that encodes both the direction and magnitude of well asymmetry:
       θ > 0 → population B well is higher → population A dominates.
       θ < 0 → population A well is higher → population B dominates.
       θ ≈ 0 → symmetric (null) landscape.
  4. Reports barrier heights ΔU_{A→B} and ΔU_{B→A}, which are proportional
     to the logarithm of the mean escape (transition) times via Kramers' law.

Assumed upstream variables
--------------------------
X_all : np.ndarray, shape (n_samples,)
    Pooled scalar signal across both populations.
X_g1  : np.ndarray, shape (n_samples_g1,)
    Scalar signal for group / population 1.
X_g2  : np.ndarray, shape (n_samples_g2,)
    Scalar signal for group / population 2.
all_results : dict
    Keyed by condition label; values are lists of dicts, each containing
    'x_grid', 'U_all', 'U_g1', 'U_g2' for one recording.
"""



GRID_POINTS = 500    # number of evaluation points for the KDE grid
BARRIER_ORDER = 15   # neighbourhood order for local extrema detection


def quasi_potential(
    data: np.ndarray,
    x_grid: np.ndarray,
    bw_method: str = "silverman",
) -> np.ndarray:
    """Estimate the quasi-potential energy landscape from a 1-D signal.

    The quasi-potential is defined as the negative log of the probability
    density estimated by a Gaussian KDE:

        U(x) = -log[P(x) + ε],   ε = 1e-10 (regularisation)

    The landscape is shifted so that its global minimum equals zero.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
        Observed 1-D signal values.
    x_grid : np.ndarray, shape (n_grid,)
        Evaluation grid for the KDE.
    bw_method : str
        Bandwidth selection method passed to ``scipy.stats.gaussian_kde``
        (default: 'silverman').

    Returns
    -------
    np.ndarray, shape (n_grid,)
        Quasi-potential U evaluated on ``x_grid``, with U_min = 0.
    """
    kde     = gaussian_kde(data, bw_method=bw_method)
    density = kde(x_grid)
    U       = -np.log(density + 1e-10)
    U      -= U.min()
    return U


def compute_dominance_angle(
    x_grid: np.ndarray,
    U: np.ndarray,
    label_a: str = "Pop A",
    label_b: str = "Pop B",
    order: int = BARRIER_ORDER,
    verbose: bool = True,
) -> dict | None:
    """Compute the tilt angle and dominance metrics of a bistable landscape.

    The two deepest local minima are identified as the two potential wells,
    and the highest local maximum between them is taken as the energy barrier.
    The dominance angle θ is defined as:

        θ = arctan(ΔU / Δx),   ΔU = U_B - U_A,  Δx = x_B - x_A > 0,

    where well A is always the left well and well B is the right well.
    A positive θ indicates that well B is higher (population A is more
    stable; population A dominates); a negative θ indicates the reverse.

    Parameters
    ----------
    x_grid : np.ndarray, shape (n_grid,)
        Evaluation grid on which U is defined.
    U : np.ndarray, shape (n_grid,)
        Quasi-potential landscape (U_min = 0).
    label_a, label_b : str
        Display names for the left and right wells.
    order : int
        Neighbourhood size (in grid points) for local extrema detection.
    verbose : bool
        If True, print a structured summary to stdout.

    Returns
    -------
    dict or None
        Dictionary with keys: x_A, U_A, x_B, U_B, x_bar, U_bar,
        delta_U, angle_deg, dominant, subdominant,
        barrier_above_A, barrier_above_B.
        Returns None if fewer than two minima or no barrier is found.
    """
    minima_idx = argrelmin(U, order=order)[0]
    maxima_idx = argrelmax(U, order=order)[0]

    if len(minima_idx) < 2:
        if verbose:
            print(f"  [{label_a} / {label_b}] Less than 2 minima — landscape may be monostable.")
        return None

    # Two deepest minima; sorted by x-position (left = A, right = B)
    sorted_minima = minima_idx[np.argsort(U[minima_idx])]
    idx_a, idx_b  = sorted(sorted_minima[:2])

    x_a, u_a = x_grid[idx_a], U[idx_a]
    x_b, u_b = x_grid[idx_b], U[idx_b]

    # Highest maximum strictly between the two minima
    between = (maxima_idx > idx_a) & (maxima_idx < idx_b)
    if between.sum() == 0:
        if verbose:
            print(f"  [{label_a} / {label_b}] No barrier found between the two minima.")
        return None

    idx_bar       = maxima_idx[between][np.argmax(U[maxima_idx[between]])]
    x_bar, u_bar  = x_grid[idx_bar], U[idx_bar]

    delta_x   = x_b - x_a          # always positive
    delta_u   = u_b - u_a          # sign encodes dominance direction
    angle_deg = np.degrees(np.arctan2(delta_u, delta_x))

    barrier_above_a = u_bar - u_a  # escape cost from well A (A → B)
    barrier_above_b = u_bar - u_b  # escape cost from well B (B → A)

    dominant    = label_a if u_a < u_b else label_b
    subdominant = label_b if u_a < u_b else label_a

    results = dict(
        x_A=x_a, U_A=u_a,
        x_B=x_b, U_B=u_b,
        x_bar=x_bar, U_bar=u_bar,
        delta_U=delta_u,
        angle_deg=angle_deg,
        dominant=dominant,
        subdominant=subdominant,
        barrier_above_A=barrier_above_a,
        barrier_above_B=barrier_above_b,
    )

    if verbose:
        print(f"  Well {label_a:<10}: x = {x_a:+.3f},  U = {u_a:.3f}")
        print(f"  Well {label_b:<10}: x = {x_b:+.3f},  U = {u_b:.3f}")
        print(f"  Barrier           : x = {x_bar:+.3f}, U = {u_bar:.3f}")
        print(f"  ΔU (B − A)        : {delta_u:+.3f}")
        print(f"  Tilt angle θ      : {angle_deg:+.2f}°")
        print(f"  Barrier A → B     : {barrier_above_a:.3f}  (escape from {label_a})")
        print(f"  Barrier B → A     : {barrier_above_b:.3f}  (escape from {label_b})")
        print(f"  ► Dominant state  : {dominant}")
        print()

    return results


def plot_dominance(
    x_grid: np.ndarray,
    U: np.ndarray,
    res: dict | None,
    title: str = "",
    label_a: str = "Pop A",
    label_b: str = "Pop B",
    color: str = "purple",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the quasi-potential landscape with dominance annotations.

    Displays the potential curve, well and barrier markers, a tilt arrow
    connecting the two wells, barrier-height brackets, and a text box
    summarising the dominance angle and dominant population.

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation grid.
    U : np.ndarray
        Quasi-potential values on ``x_grid``.
    res : dict or None
        Output of ``compute_dominance_angle``; if None, a monostable label
        is shown.
    title : str
        Axes title.
    label_a, label_b : str
        Well labels.
    color : str
        Line colour for the potential curve.
    ax : plt.Axes or None
        Target axes; a new figure is created if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_grid, U, color=color, linewidth=2.5, zorder=3)
    ax.fill_between(x_grid, U, U.max(), alpha=0.08, color=color)

    if res is None:
        ax.set_title(f"{title}\n(monostable — no angle computed)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Signal value", fontsize=10)
        ax.set_ylabel("U = −log P",   fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    # Well and barrier markers
    ax.scatter([res["x_A"]], [res["U_A"]], color="steelblue", s=100,
               zorder=5, edgecolors="k", linewidths=0.8,
               label=f"{label_a} well")
    ax.scatter([res["x_B"]], [res["U_B"]], color="tomato", s=100,
               zorder=5, edgecolors="k", linewidths=0.8,
               label=f"{label_b} well")
    ax.scatter([res["x_bar"]], [res["U_bar"]], color="gold", s=100,
               marker="^", zorder=5, edgecolors="k", linewidths=0.8,
               label="Barrier")

    # Tilt arrow: well A → well B
    ax.annotate("", xy=(res["x_B"], res["U_B"]),
                xytext=(res["x_A"], res["U_A"]),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=2, mutation_scale=18))

    # Barrier-height brackets
    for x_w, u_w, side_label, c in [
        (res["x_A"], res["U_A"],
         f"ΔU_{{A→B}}\n{res['barrier_above_A']:.2f}", "steelblue"),
        (res["x_B"], res["U_B"],
         f"ΔU_{{B→A}}\n{res['barrier_above_B']:.2f}", "tomato"),
    ]:
        ax.annotate("", xy=(x_w, res["U_bar"]), xytext=(x_w, u_w),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.5))
        mid_x = x_w + (res["x_bar"] - x_w) * 0.18
        mid_u = (u_w + res["U_bar"]) / 2
        ax.text(mid_x, mid_u, side_label,
                color=c, fontsize=8, ha="center", va="center")

    # Dominance angle annotation box
    angle_txt = (f"θ = {res['angle_deg']:+.1f}°\n"
                 f"► {res['dominant']} dominates")
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9)
    ax.text(0.97, 0.95, angle_txt, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="right",
            bbox=bbox_props)

    ax.set_xlabel("Signal value", fontsize=10)
    ax.set_ylabel("U = −log P",   fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def run_landscape_analysis(
    X_all: np.ndarray,
    X_g1: np.ndarray,
    X_g2: np.ndarray,
    label_a: str = "Pop A",
    label_b: str = "Pop B",
    grid_points: int = GRID_POINTS,
    order: int = BARRIER_ORDER,
) -> dict:
    """Compute quasi-potential landscapes and dominance angles for three groups.

    Parameters
    ----------
    X_all, X_g1, X_g2 : np.ndarray
        1-D signal arrays for the pooled, group-1, and group-2 datasets.
    label_a, label_b : str
        Population labels.
    grid_points : int
        Number of evaluation points for the KDE grid.
    order : int
        Neighbourhood size for local extrema detection.

    Returns
    -------
    dict with keys 'all', 'g1', 'g2', each mapping to a sub-dict containing
    'x_grid', 'U', and 'res' (dominance angle results).
    """
    x_min = min(X_all.min(), X_g1.min(), X_g2.min())
    x_max = max(X_all.max(), X_g1.max(), X_g2.max())
    x_grid = np.linspace(x_min, x_max, grid_points)

    output = {}
    for key, data, title in [
        ("all", X_all, "All data"),
        ("g1",  X_g1,  label_a),
        ("g2",  X_g2,  label_b),
    ]:
        print(f"── {title} ──")
        U   = quasi_potential(data, x_grid)
        res = compute_dominance_angle(x_grid, U,
                                      label_a=label_a, label_b=label_b,
                                      order=order, verbose=True)
        output[key] = dict(x_grid=x_grid, U=U, res=res, title=title)

    return output


def plot_landscape_summary(
    landscapes: dict,
    suptitle: str = "Quasi-potential landscape — dominance angle analysis",
) -> None:
    """Produce a three-panel landscape summary figure.

    Parameters
    ----------
    landscapes : dict
        Output of ``run_landscape_analysis``.
    suptitle : str
        Figure super-title.
    """
    colors = {"all": "purple", "g1": "steelblue", "g2": "tomato"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, panel) in zip(axes, landscapes.items()):
        plot_dominance(
            panel["x_grid"], panel["U"], panel["res"],
            title=panel["title"], color=colors[key], ax=ax,
        )

    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def print_condition_summary(all_results: dict) -> None:
    """Print a tabular dominance angle summary across conditions and samples.

    Parameters
    ----------
    all_results : dict
        Keyed by condition label; values are lists of dicts each with keys
        'x_grid', 'U_all', 'U_g1', 'U_g2'.
    """
    print("\n── Dominance angle summary across conditions ──")
    print(f"{'Condition':<15} {'Sample':<8} {'Group':<6} "
          f"{'Angle (°)':<12} {'Dominant':<12} {'ΔU':<10}")
    print("-" * 63)

    for condition, results_list in all_results.items():
        for s_idx, res_sample in enumerate(results_list):
            x_g = res_sample["x_grid"]
            for u_key, grp_label in [
                ("U_all", "all"),
                ("U_g1",  "g1"),
                ("U_g2",  "g2"),
            ]:
                u_s = res_sample[u_key]
                r   = compute_dominance_angle(
                    x_g, u_s, label_a="Pop A", label_b="Pop B",
                    verbose=False,
                )
                if r is not None:
                    print(f"{condition:<15} {s_idx + 1:<8} {grp_label:<6} "
                          f"{r['angle_deg']:>+8.1f}°   "
                          f"{r['dominant']:<12} "
                          f"{r['delta_U']:>+.3f}")


if __name__ == "__main__":
    # X_all, X_g1, X_g2, and all_results must be defined upstream.

    # Single-condition landscape and figure
    landscapes = run_landscape_analysis(X_all, X_g1, X_g2)
    plot_landscape_summary(landscapes)

    # Multi-condition tabular summary
    print_condition_summary(all_results)
