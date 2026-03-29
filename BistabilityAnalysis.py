import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, expon
from sklearn.mixture import GaussianMixture

"""

Bistability analysis of pituitary calcium imaging time series using
Gaussian Mixture Model (GMM) selection and dwell-time statistics.

For each recording (sample) within each physiological condition, the script:
  1. Estimates the quasi-potential landscape U(x) = -log P(x) via KDE for
     three signal pools: all cells combined, group 1 (Class 1), group 2
     (Class 2).
  2. Applies Gaussian Mixture Model (GMM) selection (1–3 components) using
     the Bayesian Information Criterion (BIC) to determine whether the
     pooled signal distribution is best described by a unimodal (n=1),
     bimodal (n=2), or trimodal (n=3) mixture.
  3. Computes per-cell dwell times in each group state and fits an
     exponential model to characterise state residence times and transition
     rates.
  4. Aggregates all metrics into a tidy summary DataFrame and produces
     two multi-panel figures:
       - Quasi-potential landscapes averaged across samples per condition.
       - Dwell-time distributions with exponential fits per condition.

Assumed upstream variables
--------------------------
data_g1 : dict
    Keys are condition labels; values are lists of np.ndarray of shape
    (n_cells_g1, n_timepoints) — one array per recording.
data_g2 : dict
    Same structure as data_g1 for group / class 2.
"""

GRID_POINTS   = 500   # KDE evaluation grid resolution
GMM_MAX_COMP  = 3     # maximum number of GMM components evaluated
GMM_N_INIT    = 10    # GMM random initialisations (guards against local minima)
RANDOM_SEED   = 42


def quasi_potential(
    data: np.ndarray,
    x_grid: np.ndarray,
    bw_method: str = "silverman",
) -> np.ndarray:
    """Estimate the quasi-potential U(x) = -log P(x) from a 1-D signal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
        Observed signal values.
    x_grid : np.ndarray, shape (n_grid,)
        Evaluation grid for the KDE.
    bw_method : str
        Bandwidth selection method for ``scipy.stats.gaussian_kde``
        (default: 'silverman').

    Returns
    -------
    np.ndarray, shape (n_grid,)
        Quasi-potential U evaluated on ``x_grid``, shifted so U_min = 0.
    """
    kde     = gaussian_kde(data, bw_method=bw_method)
    density = kde(x_grid)
    U       = -np.log(density + 1e-10)
    U      -= U.min()
    return U


def gmm_bimodality(
    data: np.ndarray,
    max_components: int = GMM_MAX_COMP,
    n_init: int = GMM_N_INIT,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[int, float], int]:
    """Select the number of Gaussian mixture components by BIC.

    Fits GMMs with 1 to ``max_components`` components and returns the BIC
    for each.  The model with the lowest BIC is selected as the best fit.
    A best-fit of n=2 is interpreted as evidence of bistability.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
        1-D signal values.
    max_components : int
        Maximum number of GMM components to evaluate.
    n_init : int
        Number of random initialisations per GMM fit (mitigates local optima).
    random_seed : int
        Random state for reproducibility.

    Returns
    -------
    bic : dict[int, float]
        BIC values keyed by number of components.
    best_n : int
        Component count with the lowest BIC.
    """
    bic = {}
    x   = data.reshape(-1, 1)
    for n in range(1, max_components + 1):
        gmm    = GaussianMixture(n_components=n, n_init=n_init,
                                 random_state=random_seed)
        gmm.fit(x)
        bic[n] = gmm.bic(x)
    best_n = min(bic, key=bic.get)
    return bic, best_n


def get_dwell_times(labels: np.ndarray) -> dict[int, list[int]]:
    """Compute run-length-encoded dwell times for each state in a label sequence.

    Parameters
    ----------
    labels : np.ndarray, shape (n_timepoints,)
        Integer state label at each time point.

    Returns
    -------
    dict[int, list[int]]
        Mapping from state label to list of consecutive run lengths (dwell times).
    """
    unique_states  = np.unique(labels)
    dwell_times    = {s: [] for s in unique_states}
    current, count = labels[0], 1

    for val in labels[1:]:
        if val == current:
            count += 1
        else:
            dwell_times[current].append(count)
            current, count = val, 1
    dwell_times[current].append(count)
    return dwell_times


def _dwell_summary(dwell_array: np.ndarray) -> dict[str, float]:
    """Return mean, median, and maximum of a dwell-time array.

    Parameters
    ----------
    dwell_array : np.ndarray
        1-D array of dwell times (may be empty).

    Returns
    -------
    dict with keys 'mean', 'median', 'max' (NaN if array is empty).
    """
    if len(dwell_array) == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan}
    return {
        "mean":   float(dwell_array.mean()),
        "median": float(np.median(dwell_array)),
        "max":    float(dwell_array.max()),
    }


def analyze_sample(
    ts1: np.ndarray,
    ts2: np.ndarray,
) -> dict:
    """Full bistability analysis for one paired recording.

    Parameters
    ----------
    ts1 : np.ndarray, shape (n_cells_g1, n_timepoints)
        Fluorescence traces for group / class 1.
    ts2 : np.ndarray, shape (n_cells_g2, n_timepoints)
        Fluorescence traces for group / class 2.

    Returns
    -------
    dict
        Contains quasi-potential arrays, GMM results, and dwell-time
        statistics.  Keys are documented in the module docstring.
    """
    x_all = np.concatenate([ts1.flatten(), ts2.flatten()])
    x_g1  = ts1.flatten()
    x_g2  = ts2.flatten()

    x_grid = np.linspace(x_all.min(), x_all.max(), GRID_POINTS)

    # Quasi-potential landscapes
    u_all = quasi_potential(x_all, x_grid)
    u_g1  = quasi_potential(x_g1,  x_grid)
    u_g2  = quasi_potential(x_g2,  x_grid)

    # GMM bimodality
    bic, best_n = gmm_bimodality(x_all)

    # Dwell times — group identity as state label
    n_cells_g1, n_timepoints = ts1.shape
    n_cells_g2, _            = ts2.shape
    n_cells_total            = n_cells_g1 + n_cells_g2

    labels = np.vstack([
        np.zeros((n_cells_g1, n_timepoints), dtype=int),
        np.ones( (n_cells_g2, n_timepoints), dtype=int),
    ])

    pooled_dwells: dict[int, list[int]] = {0: [], 1: []}
    for cell_i in range(n_cells_total):
        cell_dwells = get_dwell_times(labels[cell_i, :])
        for state in [0, 1]:
            pooled_dwells[state].extend(cell_dwells.get(state, []))

    dwell_g1 = np.array(pooled_dwells[0])
    dwell_g2 = np.array(pooled_dwells[1])

    return dict(
        x_grid=x_grid,
        U_all=u_all, U_g1=u_g1, U_g2=u_g2,
        gmm_bic=bic, gmm_best_n=best_n,
        bistable_gmm=(best_n == 2),
        dwell_g1=dwell_g1, dwell_g2=dwell_g2,
        **{f"dwell_g1_{k}": v for k, v in _dwell_summary(dwell_g1).items()},
        **{f"dwell_g2_{k}": v for k, v in _dwell_summary(dwell_g2).items()},
    )


def run_all_conditions(
    data_g1: dict,
    data_g2: dict,
) -> tuple[dict, pd.DataFrame]:
    """Run bistability analysis for all conditions and samples.

    Parameters
    ----------
    data_g1, data_g2 : dict
        Condition-keyed dictionaries of recording arrays.

    Returns
    -------
    all_results : dict
        Nested as {condition: [result_sample_1, result_sample_2, ...]}.
    summary_df : pd.DataFrame
        Tidy summary table with one row per condition × sample.
    """
    all_results: dict = {}
    rows: list[dict]  = []

    for condition in data_g1.keys():
        all_results[condition] = []
        for s_idx, (ts1, ts2) in enumerate(
            zip(data_g1[condition], data_g2[condition])
        ):
            print(f"Processing {condition} — sample {s_idx + 1}...")
            res = analyze_sample(ts1, ts2)
            all_results[condition].append(res)

            rows.append({
                "condition":       condition,
                "sample":          s_idx + 1,
                "gmm_best_n":      res["gmm_best_n"],
                "gmm_bic_1":       res["gmm_bic"][1],
                "gmm_bic_2":       res["gmm_bic"][2],
                "gmm_bic_3":       res["gmm_bic"][3],
                "bistable_gmm":    "YES" if res["bistable_gmm"] else "NO",
                "dwell_g1_mean":   res["dwell_g1_mean"],
                "dwell_g1_median": res["dwell_g1_median"],
                "dwell_g1_max":    res["dwell_g1_max"],
                "dwell_g2_mean":   res["dwell_g2_mean"],
                "dwell_g2_median": res["dwell_g2_median"],
                "dwell_g2_max":    res["dwell_g2_max"],
            })

    summary_df = pd.DataFrame(rows)
    return all_results, summary_df


def plot_landscape_conditions(all_results: dict) -> None:
    """Plot mean quasi-potential landscapes across samples per condition.

    For each condition, individual-sample landscapes are plotted as thin
    translucent lines and the cross-sample mean ± SD is shown as a thick
    line with a shaded band.

    Parameters
    ----------
    all_results : dict
        Output of ``run_all_conditions``.
    """
    conditions = list(all_results.keys())
    fig, axes  = plt.subplots(len(conditions), 3,
                               figsize=(14, 4 * len(conditions)))

    if len(conditions) == 1:
        axes = axes[np.newaxis, :]

    for row, condition in enumerate(conditions):
        results_list = all_results[condition]

        # Common grid spanning all samples in this condition
        global_min  = min(r["x_grid"].min() for r in results_list)
        global_max  = max(r["x_grid"].max() for r in results_list)
        common_grid = np.linspace(global_min, global_max, GRID_POINTS)

        for col, (u_key, color, label) in enumerate([
            ("U_all", "purple",    "All cells"),
            ("U_g1",  "steelblue", "Class 1"),
            ("U_g2",  "tomato",    "Class 2"),
        ]):
            u_stack = np.vstack([
                np.interp(common_grid, r["x_grid"], r[u_key])
                for r in results_list
            ])
            u_mean = u_stack.mean(axis=0)
            u_std  = u_stack.std(axis=0)

            ax = axes[row, col]
            for u_row in u_stack:
                ax.plot(common_grid, u_row, color=color, alpha=0.3, lw=1.0)
            ax.plot(common_grid, u_mean, color=color, lw=2.5, label="Mean")
            ax.fill_between(common_grid, u_mean - u_std, u_mean + u_std,
                            alpha=0.15, color=color, label="±1 SD")
            ax.set_title(f"{condition} — {label}", fontsize=10)
            ax.set_xlabel("Signal value", fontsize=9)
            ax.set_ylabel("U = −log P",   fontsize=9)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Quasi-potential landscapes across conditions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_dwell_distributions(all_results: dict) -> None:
    """Plot pooled dwell-time distributions with exponential fits.

    Dwell times are pooled across all samples within each condition.
    Histograms are shown on a log-density scale; an exponential distribution
    is fitted by maximum likelihood and overlaid.

    Parameters
    ----------
    all_results : dict
        Output of ``run_all_conditions``.
    """
    conditions = list(all_results.keys())
    fig, axes  = plt.subplots(len(conditions), 2,
                               figsize=(12, 4 * len(conditions)))

    if len(conditions) == 1:
        axes = axes[np.newaxis, :]

    for row, condition in enumerate(conditions):
        results_list = all_results[condition]
        pooled = {
            "g1": np.concatenate([r["dwell_g1"] for r in results_list
                                  if len(r["dwell_g1"]) > 0]),
            "g2": np.concatenate([r["dwell_g2"] for r in results_list
                                  if len(r["dwell_g2"]) > 0]),
        }

        for col, (key, color, name) in enumerate([
            ("g1", "steelblue", "Class 1 (state 0)"),
            ("g2", "tomato",    "Class 2 (state 1)"),
        ]):
            ax = axes[row, col]
            d  = pooled[key]

            if len(d) == 0:
                ax.set_title(f"{condition} — {name}\n(no transitions detected)")
                ax.spines[["top", "right"]].set_visible(False)
                continue

            bins = np.arange(1, d.max() + 2)
            ax.hist(d, bins=bins, density=True,
                    alpha=0.6, color=color, label="Observed")

            loc, scale = expon.fit(d, floc=0)
            x_fit = np.linspace(0, d.max(), 300)
            ax.plot(x_fit, expon.pdf(x_fit, loc=loc, scale=scale),
                    "k--", lw=2,
                    label=f"Exp. fit  (λ = {1.0 / scale:.3f})")

            ax.set_yscale("log")
            ax.set_xlabel("Dwell time (time points)", fontsize=9)
            ax.set_ylabel("Density (log scale)",      fontsize=9)
            ax.set_title(f"{condition} — {name}",     fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Dwell-time distributions across conditions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``run_all_conditions``.
    """
    cols = ["condition", "sample", "gmm_best_n", "bistable_gmm",
            "dwell_g1_mean", "dwell_g2_mean"]
    header = (f"{'Condition':<15} {'Sample':<8} {'GMM best n':<12} "
              f"{'Bistable':<10} {'Dwell g1 mean':<16} {'Dwell g2 mean':<14}")
    print(header)
    print("-" * len(header))
    for _, row in summary_df[cols].iterrows():
        print(f"{row['condition']:<15} {row['sample']:<8} "
              f"{row['gmm_best_n']:<12} {row['bistable_gmm']:<10} "
              f"{row['dwell_g1_mean']:<16.2f} {row['dwell_g2_mean']:<14.2f}")


if __name__ == "__main__":
    # data_g1 and data_g2 must be defined upstream (see module docstring).
    all_results, summary_df = run_all_conditions(data_g1, data_g2)

    print_summary_table(summary_df)

    # Save tidy summary to CSV
    summary_df.to_csv("bistability_summary.csv", index=False)
    print("Saved: bistability_summary.csv")

    plot_landscape_conditions(all_results)
    plot_dwell_distributions(all_results)
