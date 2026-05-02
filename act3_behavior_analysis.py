from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("yellow_tripdata_2023-01.parquet")
OUTPUT_IMAGE = Path("act3_behavior.png")
PEAK_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the columns needed for Act 3 behavior analysis."""
    required_columns = [
        "tpep_pickup_datetime",
        "trip_distance",
        "fare_amount",
    ]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_parquet(file_path, columns=required_columns)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pickup hour, peak-hour label, and efficiency features."""
    prepared = df.copy()

    prepared["tpep_pickup_datetime"] = pd.to_datetime(
        prepared["tpep_pickup_datetime"], errors="coerce"
    )
    prepared["trip_distance"] = pd.to_numeric(
        prepared["trip_distance"], errors="coerce"
    )
    prepared["fare_amount"] = pd.to_numeric(prepared["fare_amount"], errors="coerce")

    prepared = prepared.dropna(
        subset=["tpep_pickup_datetime", "trip_distance", "fare_amount"]
    )
    prepared = prepared[
        (prepared["trip_distance"] > 0) & (prepared["fare_amount"] > 0)
    ].copy()

    prepared["pickup_hour"] = prepared["tpep_pickup_datetime"].dt.hour
    prepared["is_peak_hour"] = prepared["pickup_hour"].isin(PEAK_HOURS)
    prepared["time_period"] = prepared["is_peak_hour"].map(
        {True: "Peak hours", False: "Non-peak hours"}
    )
    prepared["efficiency"] = prepared["fare_amount"] / prepared["trip_distance"]

    return prepared


def run_optional_t_test(peak_values: pd.Series, non_peak_values: pd.Series) -> None:
    """Run Welch's t-test if scipy is installed."""
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        print("\nT-test")
        print("------")
        print("SciPy is not installed, so the optional t-test was skipped.")
        return

    t_statistic, p_value = ttest_ind(
        peak_values,
        non_peak_values,
        equal_var=False,
        nan_policy="omit",
    )

    print("\nT-test")
    print("------")
    print("Test used: Welch's independent two-sample t-test")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value    : {p_value:.6f}")


def print_results(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Print hypothesis, mean comparison, difference, and interpretation."""
    peak_efficiency = df.loc[df["is_peak_hour"], "efficiency"]
    non_peak_efficiency = df.loc[~df["is_peak_hour"], "efficiency"]

    peak_mean = peak_efficiency.mean()
    non_peak_mean = non_peak_efficiency.mean()
    difference = peak_mean - non_peak_mean

    print("Act 3: Mechanical vs Behavioral Pattern Analysis")
    print("================================================")

    print("\nHypothesis")
    print("----------")
    print(
        "Trips during peak hours have higher fare per distance "
        "(efficiency) due to passenger demand."
    )

    print("\nFeature Definition")
    print("------------------")
    print("Peak hours are defined as 7-10 AM and 5-8 PM.")
    print("Efficiency is calculated as fare_amount / trip_distance.")

    print("\nDataset Used")
    print("------------")
    print(f"Rows after basic validity filtering: {len(df):,}")
    print(f"Peak-hour trips                  : {len(peak_efficiency):,}")
    print(f"Non-peak-hour trips              : {len(non_peak_efficiency):,}")

    print("\nMean Efficiency Comparison")
    print("--------------------------")
    print(f"Peak hours mean efficiency    : {peak_mean:.4f}")
    print(f"Non-peak mean efficiency      : {non_peak_mean:.4f}")
    print(f"Difference peak - non-peak    : {difference:.4f}")

    run_optional_t_test(peak_efficiency, non_peak_efficiency)

    print("\nAlternative Explanation")
    print("-----------------------")
    print(
        "A higher peak-hour efficiency may not be caused only by passenger "
        "demand. Traffic congestion can increase travel time, and taxi fares "
        "can rise when trips take longer even if the distance is similar."
    )

    print("\nInterpretation")
    print("--------------")
    if difference > 0:
        print(
            "Peak-hour trips show higher fare per distance than non-peak trips. "
            "This supports the presence of a peak-hour pattern, but the stronger "
            "explanation is mechanical: congestion and time-based fare components "
            "can raise fare per mile without proving a direct change in passenger "
            "behavior."
        )
    elif difference < 0:
        print(
            "Peak-hour trips show lower fare per distance than non-peak trips. "
            "This weakens the demand-based behavioral hypothesis and suggests "
            "that trip mix, routing, or fare structure may explain the pattern."
        )
    else:
        print(
            "Peak and non-peak trips have nearly identical efficiency. The data "
            "does not provide strong evidence for a behavioral or mechanical "
            "difference based on this measure alone."
        )

    print("\nConclusion")
    print("----------")
    print(
        "The observed pattern should not automatically be treated as human "
        "behavior. Fare per distance can be shaped by mechanical system factors "
        "such as congestion, travel time, and fare rules. Therefore, the "
        "mechanical explanation is stronger unless additional evidence directly "
        "measures passenger demand."
    )

    return peak_efficiency, non_peak_efficiency


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart comparing mean efficiency by time period."""
    summary = (
        df.groupby("time_period")["efficiency"]
        .mean()
        .reindex(["Peak hours", "Non-peak hours"])
    )

    ax = summary.plot(
        kind="bar",
        figsize=(8, 5),
        color=["tomato", "steelblue"],
        edgecolor="black",
    )
    ax.set_title("Mean Fare Efficiency: Peak vs Non-Peak Hours")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Mean Efficiency (fare per mile)")
    ax.set_xticklabels(summary.index, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    raw_df = load_dataset(DATA_FILE)
    analysis_df = prepare_features(raw_df)
    print_results(analysis_df)
    plot_comparison(analysis_df, OUTPUT_IMAGE)
    print(f"\nVisualization saved as: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
