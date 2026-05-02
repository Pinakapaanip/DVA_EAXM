from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("yellow_tripdata_2023-01.parquet")
OUTPUT_IMAGE = Path("act5_policy.png")
R_PERCENT = 8
ZONE_ORDER = ["Stable", "Stress", "Unstable"]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the columns required for the policy simulation."""
    required_columns = [
        "trip_distance",
        "fare_amount",
        "tpep_pickup_datetime",
    ]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_parquet(file_path, columns=required_columns)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records that cannot support fare-per-distance analysis."""
    cleaned = df.copy()

    cleaned["tpep_pickup_datetime"] = pd.to_datetime(
        cleaned["tpep_pickup_datetime"], errors="coerce"
    )
    cleaned["trip_distance"] = pd.to_numeric(
        cleaned["trip_distance"], errors="coerce"
    )
    cleaned["fare_amount"] = pd.to_numeric(cleaned["fare_amount"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=["tpep_pickup_datetime", "trip_distance", "fare_amount"]
    )
    cleaned = cleaned[
        (cleaned["trip_distance"] > 0) & (cleaned["fare_amount"] > 0)
    ].copy()

    return cleaned


def classify_zone(efficiency: pd.Series) -> pd.Series:
    """Classify efficiency values into Stable, Stress, and Unstable zones."""
    zones = pd.Series("Unstable", index=efficiency.index)
    zones.loc[efficiency < 7] = "Stable"
    zones.loc[efficiency.between(7, 9, inclusive="both")] = "Stress"
    zones.loc[efficiency > 9] = "Unstable"
    return zones


def simulate_policy(df: pd.DataFrame, r_percent: int) -> pd.DataFrame:
    """Apply an r percent fare increase and recompute efficiency and zones."""
    simulated = df.copy()
    multiplier = 1 + (r_percent / 100)

    simulated["efficiency"] = (
        simulated["fare_amount"] / simulated["trip_distance"]
    )
    simulated["zone_before"] = classify_zone(simulated["efficiency"])

    simulated["fare_modified"] = simulated["fare_amount"] * multiplier
    simulated["efficiency_new"] = (
        simulated["fare_modified"] / simulated["trip_distance"]
    )
    simulated["zone_after"] = classify_zone(simulated["efficiency_new"])

    return simulated


def zone_counts(zone_series: pd.Series) -> pd.Series:
    """Return zone counts in a fixed order."""
    return zone_series.value_counts().reindex(ZONE_ORDER, fill_value=0)


def print_results(
    raw_rows: int,
    cleaned_rows: int,
    before_counts: pd.Series,
    after_counts: pd.Series,
    r_percent: int,
) -> None:
    """Print the policy simulation results and interpretation."""
    comparison = pd.DataFrame(
        {
            "Before policy": before_counts,
            "After policy": after_counts,
        }
    )
    comparison["Change"] = comparison["After policy"] - comparison["Before policy"]
    comparison["Percent change"] = (
        comparison["Change"] / comparison["Before policy"].replace(0, pd.NA)
    ) * 100

    stable_change = int(comparison.loc["Stable", "Change"])
    stress_change = int(comparison.loc["Stress", "Change"])
    unstable_change = int(comparison.loc["Unstable", "Change"])

    print("Act 5: Policy Simulation")
    print("========================")

    print("\nPolicy Parameter")
    print("----------------")
    print(f"r = {r_percent}")
    print(f"Policy simulated: fare_amount increased by {r_percent} percent.")

    print("\nData Cleaning")
    print("-------------")
    print(f"Rows before cleaning: {raw_rows:,}")
    print(f"Rows after cleaning : {cleaned_rows:,}")
    print(f"Rows removed        : {raw_rows - cleaned_rows:,}")

    print("\nBaseline Metric")
    print("---------------")
    print("efficiency = fare_amount / trip_distance")
    print("Zones: <7 Stable, 7-9 Stress, >9 Unstable")

    print("\nZone Distribution Before Policy")
    print("-------------------------------")
    print(before_counts)

    print("\nZone Distribution After Policy")
    print("------------------------------")
    print(after_counts)

    print("\nZone Distribution Change")
    print("------------------------")
    print(comparison)

    print("\nInterpretation")
    print("--------------")
    if stress_change > 0 or unstable_change > 0:
        print(
            "The fare increase moves some trips into higher efficiency zones. "
            "This means the system does not absorb the policy change completely; "
            "stress or instability increases for part of the dataset."
        )
    else:
        print(
            "The fare increase does not increase Stress or Unstable counts. "
            "Under these thresholds, the system absorbs the change without a "
            "visible zone-level shift."
        )

    if unstable_change > stress_change:
        print(
            "The effect is amplified near the upper threshold because more trips "
            "cross into the Unstable zone than into the Stress zone."
        )
    elif stable_change < 0 and (stress_change > 0 or unstable_change > 0):
        print(
            "The effect is threshold-driven rather than perfectly proportional: "
            "an 8 percent fare increase can cause larger category changes for "
            "trips already close to zone boundaries."
        )
    else:
        print(
            "The effect is mostly proportional in the metric, with limited "
            "amplification in zone membership."
        )

    print(
        "Because the thresholds are fixed, even a simple percentage change can "
        "alter the apparent behavior of the taxi system. Policy conclusions "
        "therefore depend on both the parameter value and the zone definition."
    )


def plot_policy_comparison(
    before_counts: pd.Series, after_counts: pd.Series, output_path: Path
) -> None:
    """Save a side-by-side bar chart of zone counts before and after policy."""
    comparison = pd.DataFrame(
        {
            "Before policy": before_counts,
            "After policy": after_counts,
        }
    )

    ax = comparison.plot(
        kind="bar",
        figsize=(9, 5),
        color=["steelblue", "tomato"],
        edgecolor="black",
    )
    ax.set_title("Zone Distribution Before vs After 8 Percent Fare Increase")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Number of Trips")
    ax.set_xticklabels(ZONE_ORDER, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    raw_df = load_dataset(DATA_FILE)
    cleaned_df = clean_dataset(raw_df)
    simulated_df = simulate_policy(cleaned_df, R_PERCENT)

    before_counts = zone_counts(simulated_df["zone_before"])
    after_counts = zone_counts(simulated_df["zone_after"])

    print_results(
        raw_rows=len(raw_df),
        cleaned_rows=len(cleaned_df),
        before_counts=before_counts,
        after_counts=after_counts,
        r_percent=R_PERCENT,
    )

    plot_policy_comparison(before_counts, after_counts, OUTPUT_IMAGE)
    print(f"\nPolicy comparison plot saved as: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
