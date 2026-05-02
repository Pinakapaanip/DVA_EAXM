from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("yellow_tripdata_2023-01.parquet")
OUTPUT_IMAGE = Path("act2_comparison.png")
ZONE_ORDER = ["Stable", "Stress", "Unstable"]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the required columns from the NYC Yellow Taxi parquet file."""
    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
    ]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_parquet(file_path, columns=required_columns)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to suitable data types and calculate trip duration."""
    prepared = df.copy()

    prepared["tpep_pickup_datetime"] = pd.to_datetime(
        prepared["tpep_pickup_datetime"], errors="coerce"
    )
    prepared["tpep_dropoff_datetime"] = pd.to_datetime(
        prepared["tpep_dropoff_datetime"], errors="coerce"
    )
    prepared["trip_distance"] = pd.to_numeric(
        prepared["trip_distance"], errors="coerce"
    )
    prepared["fare_amount"] = pd.to_numeric(prepared["fare_amount"], errors="coerce")

    trip_duration = (
        prepared["tpep_dropoff_datetime"] - prepared["tpep_pickup_datetime"]
    )
    prepared["duration_minutes"] = trip_duration.dt.total_seconds() / 60

    return prepared


def detect_invalid_trips(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Detect invalid trips using the Act 2 validation conditions."""
    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
    ]

    invalid_conditions = {
        "Missing values": df[required_columns].isna().any(axis=1),
        "trip_distance <= 0": df["trip_distance"] <= 0,
        "fare_amount <= 0": df["fare_amount"] <= 0,
        "Invalid duration pickup > dropoff": (
            df["tpep_pickup_datetime"] > df["tpep_dropoff_datetime"]
        ),
    }

    invalid_conditions["Any invalid condition"] = (
        invalid_conditions["Missing values"]
        | invalid_conditions["trip_distance <= 0"]
        | invalid_conditions["fare_amount <= 0"]
        | invalid_conditions["Invalid duration pickup > dropoff"]
    )

    return invalid_conditions


def print_invalid_counts(invalid_conditions: dict[str, pd.Series]) -> None:
    """Print the count of each invalid condition."""
    print("\nInvalid Trip Counts")
    print("-------------------")
    for condition, mask in invalid_conditions.items():
        print(f"{condition:35s}: {int(mask.sum()):,}")


def clean_real_trips(
    df: pd.DataFrame, invalid_conditions: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Keep only real trips.

    A real trip is defined as a complete trip record where:
    - pickup and dropoff timestamps are present
    - trip distance is greater than 0
    - fare amount is greater than 0
    - pickup time is not after dropoff time
    """
    real_trip_mask = ~invalid_conditions["Any invalid condition"]
    return df.loc[real_trip_mask].copy()


def add_efficiency_and_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute efficiency and classify each trip into a zone."""
    result = df.copy()
    result["efficiency"] = result["fare_amount"] / result["trip_distance"]

    result["zone"] = "Unstable"
    result.loc[result["efficiency"] < 7, "zone"] = "Stable"
    result.loc[result["efficiency"].between(7, 9, inclusive="both"), "zone"] = (
        "Stress"
    )
    result.loc[result["efficiency"] > 9, "zone"] = "Unstable"

    return result


def get_zone_distribution(df: pd.DataFrame) -> pd.Series:
    """Return zone counts in a consistent order."""
    return df["zone"].value_counts().reindex(ZONE_ORDER, fill_value=0)


def plot_zone_comparison(
    before_counts: pd.Series, after_counts: pd.Series, output_path: Path
) -> None:
    """Create and save a side-by-side bar chart for zone distributions."""
    comparison = pd.DataFrame(
        {
            "Before cleaning": before_counts,
            "After cleaning": after_counts,
        }
    )

    ax = comparison.plot(
        kind="bar",
        figsize=(9, 5),
        color=["steelblue", "seagreen"],
        edgecolor="black",
    )
    ax.set_title("Zone Distribution Before vs After Data Cleaning")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Number of Trips")
    ax.set_xticklabels(ZONE_ORDER, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    print("Act 2: Data Validation and Cleaning")
    print("===================================")

    raw_df = load_dataset(DATA_FILE)
    prepared_df = prepare_dataset(raw_df)

    invalid_conditions = detect_invalid_trips(prepared_df)
    print_invalid_counts(invalid_conditions)

    before_cleaning = add_efficiency_and_zone(prepared_df)
    cleaned_df = clean_real_trips(prepared_df, invalid_conditions)
    after_cleaning = add_efficiency_and_zone(cleaned_df)

    print("\nReal Trip Definition")
    print("--------------------")
    print(
        "A real trip has no missing required values, trip_distance > 0, "
        "fare_amount > 0, and pickup time not after dropoff time."
    )

    print("\nDataset Size")
    print("------------")
    print(f"Before cleaning: {len(prepared_df):,} rows")
    print(f"After cleaning : {len(after_cleaning):,} rows")
    print(f"Rows removed   : {len(prepared_df) - len(after_cleaning):,} rows")

    before_counts = get_zone_distribution(before_cleaning)
    after_counts = get_zone_distribution(after_cleaning)

    print("\nZone Distribution Before Cleaning")
    print("---------------------------------")
    print(before_counts)

    print("\nZone Distribution After Cleaning")
    print("--------------------------------")
    print(after_counts)

    plot_zone_comparison(before_counts, after_counts, OUTPUT_IMAGE)
    print(f"\nSide-by-side comparison plot saved as: {OUTPUT_IMAGE}")

    print("\nConclusion")
    print("----------")
    print(
        "Defining what counts as a real trip changes the interpretation of "
        "system behavior. Invalid records can distort efficiency and make the "
        "taxi system appear more stable, stressed, or unstable than it really "
        "is. Cleaning the data ensures that the final zone distribution is "
        "based on meaningful trip records rather than data-entry errors or "
        "contradictory values."
    )


if __name__ == "__main__":
    main()
