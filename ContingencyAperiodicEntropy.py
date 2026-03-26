
import numpy as np
import pandas as pd

"""

Contingency table analysis of the joint distribution of aperiodic exponent
and Shannon entropy (H) across spectral clusters, cell populations, and
physiological conditions.

Both continuous measures are discretised into tertiles (Low / Medium / High)
before cross-tabulation.  Marginal and joint contingency tables are computed
at three levels of granularity:

    1. All populations and conditions pooled.
    2. A single cell population, all conditions.
    3. A single cell population and a single condition.

Percentage representations of the joint table are saved to CSV for
downstream figure generation.

Input file
----------
EntropyOfTheSignal.csv : semicolon-delimited table with at least the columns
    CellGroup, Condition, AperiodicValue, H, Cluster.
"""



CONDITIONS  = ["Virgin", "Lactant", "Multipara", "Weaned", "OVX"]
POPULATIONS = ["Lactotrophs", "Somatotrophs", "All population"]
TERTILE_LABELS = ["Low", "Medium", "High"]


def load_data(path: str) -> pd.DataFrame:
    """Load and column-strip the input CSV file.

    Parameters
    ----------
    path : str
        Path to the semicolon-delimited CSV.

    Returns
    -------
    pd.DataFrame
        Table with stripped column names and original dtypes preserved.
    """
    df = pd.read_csv(path, delimiter=";")
    df.columns = df.columns.str.strip()
    return df


def discretise_tertiles(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Bin continuous columns into population tertiles in-place.

    Each column is independently cut at its 33rd and 67th percentiles so that
    approximately one-third of observations fall in each bin.  The original
    continuous columns are replaced by ordered categoricals.

    Parameters
    ----------
    df : pd.DataFrame
        Input table (modified in-place).
    columns : list of str
        Column names to discretise.
    labels : list of str or None
        Bin labels; defaults to ['Low', 'Medium', 'High'].

    Returns
    -------
    pd.DataFrame
        Table with specified columns replaced by tertile categoricals.
    """
    if labels is None:
        labels = TERTILE_LABELS

    df = df.copy()
    for col in columns:
        df[col] = pd.qcut(df[col], q=3, labels=labels)
    return df


def contingency_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute marginal and joint contingency tables against cluster identity.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the dataset (already filtered to the desired scope).
        Must contain columns AperiodicValue, H, Cluster.

    Returns
    -------
    aperiodic_table : pd.DataFrame
        Cross-tabulation of AperiodicValue (rows) by Cluster (columns).
    entropy_table : pd.DataFrame
        Cross-tabulation of H (rows) by Cluster (columns).
    joint_table : pd.DataFrame
        Cross-tabulation of (AperiodicValue, H) jointly by Cluster.
    """
    aperiodic_table = pd.crosstab(df["AperiodicValue"], df["Cluster"])
    entropy_table   = pd.crosstab(df["H"],              df["Cluster"])
    joint_table     = pd.crosstab([df["AperiodicValue"], df["H"]], df["Cluster"])
    return aperiodic_table, entropy_table, joint_table


def joint_table_percentage(
    df: pd.DataFrame,
    joint_table: pd.DataFrame,
) -> pd.DataFrame:
    """Express joint contingency counts as percentage of total observations.

    Parameters
    ----------
    df : pd.DataFrame
        The subset used to build ``joint_table`` (provides the denominator).
    joint_table : pd.DataFrame
        Absolute-count joint contingency table.

    Returns
    -------
    pd.DataFrame
        Joint table with entries expressed as percentage of total cell count.
    """
    n_total = len(df)
    return (joint_table * 100) / n_total


def run_contingency_analysis(
    df: pd.DataFrame,
    populations: list[str],
    conditions: list[str],
    output_dir: str = ".",
) -> None:
    """Run the full contingency analysis across all population/condition pairs.

    For each combination, three tables are computed (marginal aperiodic,
    marginal entropy, joint) and the percentage joint table is saved to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Full discretised dataset.
    populations : list of str
        Cell population labels to iterate over.
    conditions : list of str
        Physiological condition labels to iterate over.
    output_dir : str
        Directory in which to save output CSV files (default: current dir).
    """
    import os

    # --- Level 1: pooled (all populations, all conditions) ---
    ap_table, ent_table, jt = contingency_tables(df)
    pct_jt = joint_table_percentage(df, jt)
    out_path = os.path.join(output_dir, "joint_table_all.csv")
    pct_jt.to_csv(out_path)
    print(f"Saved: {out_path}")

    for population in populations:
        pop_df = df[df["CellGroup"] == population]

        if pop_df.empty:
            print(f"Skipping population '{population}': no data.")
            continue

        # --- Level 2: single population, all conditions pooled ---
        ap_table, ent_table, jt = contingency_tables(pop_df)
        pct_jt = joint_table_percentage(pop_df, jt)
        out_path = os.path.join(output_dir, f"joint_table_{population}.csv")
        pct_jt.to_csv(out_path)
        print(f"Saved: {out_path}")

        for condition in conditions:
            cond_df = pop_df[pop_df["Condition"] == condition]

            if cond_df.empty:
                print(f"  Skipping {population} / {condition}: no data.")
                continue

            # --- Level 3: single population × single condition ---
            ap_table, ent_table, jt = contingency_tables(cond_df)
            pct_jt = joint_table_percentage(cond_df, jt)

            label = f"{population}_{condition}".replace(" ", "_")
            out_path = os.path.join(output_dir, f"joint_table_{label}.csv")
            pct_jt.to_csv(out_path)
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    df_raw = load_data("EntropyOfTheSignal.csv")

    df = discretise_tertiles(df_raw, columns=["AperiodicValue", "H"])

    run_contingency_analysis(
        df,
        populations=POPULATIONS,
        conditions=CONDITIONS,
        output_dir=".",
    )
