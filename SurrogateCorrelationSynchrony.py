
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

plt.style.use("fivethirtyeight")

"""

Two-stage analysis of pairwise cell interactions from calcium fluorescence
time series:

  Stage 1 — Surrogate-thresholded Spearman correlation matrix.
    A Fourier phase-randomisation surrogate procedure (N = 1,000 resamples)
    defines a significance threshold at mean + 2 SD of the null distribution.
    Observed pairwise Spearman correlations that do not exceed this threshold
    are zeroed, and the surviving interactions are classified as strongly
    synchronous, weakly asynchronous, or negatively asynchronous.

  Stage 2 — Cluster–synchrony association via Random Forest classification.
    Spectral cluster labels and synchrony labels are merged at the cell level,
    and a Random Forest classifier is evaluated by 5-fold cross-validation to
    quantify how well cluster identity predicts synchrony class.

Input files
-----------
ClusterINDEX.csv  : semicolon-delimited; columns Cells, Population, Condition, Cluster.
SynAsynINDEX.csv  : semicolon-delimited; columns Cells, Population, Condition, Kind. ## structured since the Stage 1

Assumed upstream variable
--------------------------
datosNorm_exponential : np.ndarray, shape (n_cells, n_samples)
    Drift-corrected, low-pass-filtered fluorescence traces from preprocessing.
"""


N_SAMPLES   = 300          # number of time points used for correlation
N_SURROGATE = 1_000        # number of Fourier surrogate resamples
THRESHOLD_SD_FACTOR = 2.0  # threshold = mean + THRESHOLD_SD_FACTOR * SD
RANDOM_SEED = 42


def fourier_surrogate_correlation(
    data: np.ndarray,
    n_samples: int,
    n_surrogate: int = 1_000,
) -> np.ndarray:
    """Generate surrogate Spearman correlation matrices via Fourier phase randomisation.

    For each resample, the FFT phase spectrum of every cell is independently
    replaced by uniform random phases while preserving the amplitude spectrum,
    enforcing conjugate symmetry so that the inverse FFT yields a real-valued
    signal.  The Spearman correlation matrix of the surrogate data is computed
    and stored.

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_samples_full)
        Fluorescence time series; only the first ``n_samples`` columns are used.
    n_samples : int
        Number of time points to retain before surrogate generation.
    n_surrogate : int
        Number of surrogate resamples (default 1,000).

    Returns
    -------
    np.ndarray, shape (n_surrogate, n_cells, n_cells)
        Stack of surrogate Spearman correlation matrices.
    """
    data = data[:, :n_samples]
    n_half = n_samples // 2

    fft_data = np.fft.fft(data, axis=-1)
    amplitude = np.abs(fft_data)
    phase     = np.angle(fft_data)

    surrogate_matrices = []

    for _ in range(n_surrogate):
        phase_surr = np.random.uniform(-np.pi, np.pi, size=phase.shape)
        # Enforce conjugate symmetry for real-valued IFFT output
        phase_surr[:, n_half:] = -phase_surr[:, n_half:0:-1]
        phase_surr[:, n_half]  = 0.0

        fft_surr  = amplitude * (np.cos(phase_surr) + 1j * np.sin(phase_surr))
        data_surr = np.real(np.fft.ifft(fft_surr, axis=-1))

        corr_surr, _ = stats.spearmanr(data_surr, axis=1)
        surrogate_matrices.append(corr_surr)

    return np.array(surrogate_matrices)


def threshold_correlation_matrix(
    observed: np.ndarray,
    surrogate_mean: np.ndarray,
    surrogate_sd: np.ndarray,
    sd_factor: float = 2.0,
) -> np.ndarray:
    """Zero out observed correlations that fall below the surrogate threshold.

    An observed correlation is retained only if its absolute value exceeds
    ``surrogate_mean + sd_factor * surrogate_sd``.

    Parameters
    ----------
    observed : np.ndarray, shape (n_cells, n_cells)
        Observed Spearman correlation matrix.
    surrogate_mean : np.ndarray, shape (n_cells, n_cells)
        Element-wise mean of the surrogate distribution.
    surrogate_sd : np.ndarray, shape (n_cells, n_cells)
        Element-wise standard deviation of the surrogate distribution.
    sd_factor : float
        Multiplier applied to ``surrogate_sd`` (default 2.0).

    Returns
    -------
    np.ndarray, shape (n_cells, n_cells)
        Thresholded correlation matrix with sub-threshold entries set to zero.
    """
    threshold = surrogate_mean + sd_factor * surrogate_sd
    thresholded = observed.copy()
    thresholded[np.abs(thresholded) < threshold] = 0.0
    return thresholded


def classify_interactions(
    corr: np.ndarray,
    surrogate_mean: np.ndarray,
    surrogate_sd: np.ndarray,
    sd_factor: float = 2.0,
) -> dict[str, np.ndarray]:
    """Partition non-zero correlations into three interaction classes.

    Classes
    -------
    strong_synchronous   : corr > mean + sd_factor * SD
    negative_asynchronous: corr < -(mean) and corr != 0
    weak_asynchronous    : |corr| < mean and corr != 0

    Parameters
    ----------
    corr : np.ndarray, shape (n_cells, n_cells)
        Thresholded Spearman correlation matrix.
    surrogate_mean : np.ndarray
        Element-wise surrogate mean.
    surrogate_sd : np.ndarray
        Element-wise surrogate SD.
    sd_factor : float
        Multiplier for the strong synchrony threshold (default 2.0).

    Returns
    -------
    dict with keys 'strong_synchronous', 'negative_asynchronous',
    'weak_asynchronous', each mapping to a 1-D array of correlation values.
    """
    threshold_high = surrogate_mean + sd_factor * surrogate_sd

    strong_sync_mask   = corr > threshold_high
    neg_async_mask     = (corr < -surrogate_mean) & (corr != 0)
    weak_async_mask    = (np.abs(corr) < surrogate_mean) & (corr != 0)

    return {
        "strong_synchronous":    corr[strong_sync_mask].flatten(),
        "negative_asynchronous": corr[neg_async_mask].flatten(),
        "weak_asynchronous":     corr[weak_async_mask].flatten(),
    }


def plot_correlation_matrix(corr: np.ndarray, title: str = "Spearman Correlation Matrix") -> None:
    """Plot a correlation matrix as a heatmap.

    Parameters
    ----------
    corr : np.ndarray, shape (n_cells, n_cells)
        Correlation matrix to visualise.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="seismic", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.grid(False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_interaction_boxplot(interaction_values: dict[str, np.ndarray]) -> None:
    """Plot a boxplot of correlation values for each interaction class.

    Parameters
    ----------
    interaction_values : dict
        Keys are class labels; values are 1-D arrays of correlation values.
    """
    labels = list(interaction_values.keys())
    data   = [interaction_values[k] for k in labels]
    display_labels = ["Strong synchronous", "Negative asynchronous", "Weak asynchronous"]

    fig, ax = plt.subplots()
    ax.boxplot(
        data,
        labels=display_labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )
    ax.set_title("Distribution of thresholded Spearman correlations")
    ax.set_ylabel("Correlation value")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def load_and_merge_labels(
    cluster_path: str,
    synasynpath: str,
) -> pd.DataFrame:
    """Load cluster and synchrony label files and merge on shared keys.

    Parameters
    ----------
    cluster_path : str
        Path to ClusterINDEX.csv.
    synasynpath : str
        Path to SynAsynINDEX.csv.

    Returns
    -------
    pd.DataFrame
        Inner-joined table with columns Cells, Population, Condition,
        Cluster, Kind.
    """
    df_clusters = pd.read_csv(cluster_path, delimiter=";")
    df_kind     = pd.read_csv(synasynpath,  delimiter=";")

    df_clusters.columns = df_clusters.columns.str.strip()
    df_kind.columns     = df_kind.columns.str.strip()

    return pd.merge(df_clusters, df_kind,
                    on=["Cells", "Population", "Condition"], how="inner")


def encode_synchrony_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Recode synchrony string labels to integers (Sync → 1, Async → 2).

    Trailing whitespace variants (e.g. 'Sync ') are handled before encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Merged label table with a 'Kind' column.

    Returns
    -------
    pd.DataFrame
        Table with 'Kind' replaced by integer codes.
    """
    df = df.copy()
    df["Kind"] = df["Kind"].str.strip().map({"Sync": 1, "Async": 2})
    return df.dropna(subset=["Kind"])


def random_forest_synchrony(
    df: pd.DataFrame,
    population: str,
    condition: str,
    n_folds: int = 5,
    random_seed: int = 42,
) -> None:
    """Fit and cross-validate a Random Forest classifier for synchrony prediction.

    Cluster identity is used as the sole feature; synchrony label (Sync / Async)
    is the target.  A confusion matrix and 5-fold cross-validated accuracy are
    reported.

    Parameters
    ----------
    df : pd.DataFrame
        Merged, encoded label table.
    population : str
        Population to filter (e.g. 'All population').
    condition : str
        Condition to filter (e.g. 'Multipara').
    n_folds : int
        Number of cross-validation folds (default 5).
    random_seed : int
        Random state for the classifier (default 42).
    """
    subset = df[
        (df["Population"] == population) &
        (df["Condition"]  == condition)
    ].dropna(subset=["Cluster", "Kind"])

    if len(subset) < n_folds:
        print(f"Insufficient data for {population} / {condition}. Skipping.")
        return

    X = subset[["Cluster"]].values
    y = subset["Kind"].values

    cm = confusion_matrix(y, X)  # note: compare predicted (cluster) vs true (kind)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sync", "Async"],
                yticklabels=["Cluster 1", "Cluster 2"],
                vmin=0, vmax=cm.max(), ax=ax)
    ax.set_title(f"Confusion matrix — {population}, {condition}")
    ax.set_xlabel("Predicted (cluster)")
    ax.set_ylabel("Actual (synchrony)")
    plt.tight_layout()
    plt.show()

    clf    = RandomForestClassifier(random_state=random_seed)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
    print(
        f"{population} / {condition}: "
        f"Cross-validated accuracy = {scores.mean():.3f} (±{scores.std():.3f})"
    )


if __name__ == "__main__":
    # --- Stage 1: surrogate-thresholded Spearman correlation ---
    surrogate_matrices = fourier_surrogate_correlation(
        datosNorm_exponential, n_samples=N_SAMPLES, n_surrogate=N_SURROGATE
    )

    surrogate_mean = np.mean(surrogate_matrices, axis=0)
    surrogate_sd   = np.std(surrogate_matrices,  axis=0)

    observed_corr, _ = stats.spearmanr(
        datosNorm_exponential[:, :N_SAMPLES], axis=1
    )

    corr_thresholded = threshold_correlation_matrix(
        observed_corr, surrogate_mean, surrogate_sd, sd_factor=THRESHOLD_SD_FACTOR
    )

    plot_correlation_matrix(corr_thresholded)

    interactions = classify_interactions(
        corr_thresholded, surrogate_mean, surrogate_sd, sd_factor=THRESHOLD_SD_FACTOR
    )
    plot_interaction_boxplot(interactions)

    positive     = np.sum(corr_thresholded > 0)
    negative     = np.sum(corr_thresholded < 0)
    total_nonzero = positive + negative

    print(f"Positive interactions : {positive}  ({positive / total_nonzero:.3f})")
    print(f"Negative interactions : {negative}  ({negative / total_nonzero:.3f})")

    # --- Stage 2: cluster–synchrony classification ---
    df_merged = load_and_merge_labels("ClusterINDEX.csv", "SynAsynINDEX.csv")
    print(pd.crosstab(df_merged["Cluster"], df_merged["Kind"]))

    df_encoded = encode_synchrony_labels(df_merged)

    random_forest_synchrony(
        df_encoded,
        population="All population",
        condition="Multipara",
        n_folds=5,
        random_seed=RANDOM_SEED,
    )
