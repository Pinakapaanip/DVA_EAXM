import importlib
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "pyarrow": "pyarrow",
}

DATA_FILE = Path("yellow_tripdata_2023-01.parquet")
OUTPUT_PLOT = Path("zone_plot.png")
ZONE_ORDER = ["Stable", "Stress", "Unstable"]


def install_missing_packages() -> None:
    """Install required packages that are not already available."""
    print("Checking required libraries...")

    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
            print(f"  {package_name} is already installed.")
        except ImportError:
            print(f"  {package_name} is missing. Installing...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name]
                )
                print(f"  {package_name} installed successfully.")
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Failed to install {package_name}. "
                    "Please install it manually and run the script again."
                ) from error


def load_data(file_path: Path):
    """Load only the columns required for validation and analysis."""
    import pandas as pd

    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
    ]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Make sure it is in the current directory."
        )

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_parquet(file_path, columns=required_columns)
    except Exception as error:
        raise RuntimeError(f"Failed to load parquet file: {error}") from error

    print(f"Loaded {len(df):,} raw rows.")
    return df


def prepare_columns(df):
    """Convert fields to analysis-friendly types and calculate duration."""
    import pandas as pd

    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"], errors="coerce"
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"], errors="coerce"
    )
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")

    duration = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["duration_minutes"] = duration.dt.total_seconds() / 60
    return df


def add_efficiency_and_zone(df):
    """Calculate fare per mile and classify each trip into a behavior zone."""
    import numpy as np

    df = df.copy()
    df["efficiency"] = df["fare_amount"] / df["trip_distance"]

    conditions = [
        df["efficiency"] < 7,
        df["efficiency"].between(7, 9, inclusive="both"),
        df["efficiency"] > 9,
    ]
    choices = ["Stable", "Stress", "Unstable"]
    df["zone"] = np.select(conditions, choices, default="Unstable")
    return df


def detect_invalid_trips(df):
    """Return invalid-trip masks for each rule used in Act 2."""
    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "duration_minutes",
    ]

    missing_values = df[required_columns].isna().any(axis=1)
    non_positive_distance = df["trip_distance"] <= 0
    non_positive_fare = df["fare_amount"] <= 0
    pickup_after_dropoff = df["tpep_pickup_datetime"] > df["tpep_dropoff_datetime"]
    non_positive_duration = df["duration_minutes"] <= 0

    # Domain limits remove trips that are mathematically possible but unrealistic
    # for ordinary NYC yellow taxi behavior.
    extreme_outlier = (
        (df["duration_minutes"] > 24 * 60)
        | (df["trip_distance"] > 100)
        | (df["fare_amount"] > 500)
    )

    possible_efficiency = df["fare_amount"] / df["trip_distance"]
    extreme_outlier = extreme_outlier | (possible_efficiency > 100)

    invalid_masks = {
        "Missing values": missing_values,
        "trip_distance <= 0": non_positive_distance,
        "fare_amount <= 0": non_positive_fare,
        "Pickup time after dropoff time": pickup_after_dropoff,
        "Duration <= 0 minutes": non_positive_duration,
        "Extreme outlier": extreme_outlier,
    }
    invalid_masks["Any invalid rule"] = (
        missing_values
        | non_positive_distance
        | non_positive_fare
        | pickup_after_dropoff
        | non_positive_duration
        | extreme_outlier
    )
    return invalid_masks


def clean_data(df, invalid_masks):
    """Keep only rows that satisfy the real-trip definition."""
    real_trip_mask = ~invalid_masks["Any invalid rule"]
    return df.loc[real_trip_mask].copy()


def zone_counts(df):
    """Return zone counts in a consistent order."""
    return df["zone"].value_counts().reindex(ZONE_ORDER, fill_value=0)


def plot_zone_comparison(before_counts, after_counts, output_path: Path) -> None:
    """Save a side-by-side bar chart comparing raw and cleaned zone counts."""
    import matplotlib.pyplot as plt
    import pandas as pd

    comparison = pd.DataFrame(
        {
            "Before cleaning": before_counts,
            "After cleaning": after_counts,
        }
    )

    ax = comparison.plot(
        kind="bar",
        figsize=(9, 5),
        color=["#5b8def", "#2f9e44"],
        edgecolor="black",
    )
    ax.set_title("NYC Yellow Taxi Zone Distribution Before and After Cleaning")
    ax.set_xlabel("Efficiency Zone")
    ax.set_ylabel("Number of Trips")
    ax.set_xticklabels(ZONE_ORDER, rotation=0)
    ax.legend(title="")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nPlot saved successfully as {output_path}")


def print_invalid_trip_report(df, invalid_masks) -> None:
    """Print invalid-trip counts and dataset-size summary."""
    total_rows = len(df)
    print("\nInvalid Trip Detection")
    print("----------------------")
    for rule_name, mask in invalid_masks.items():
        count = int(mask.sum())
        percentage = (count / total_rows) * 100 if total_rows else 0
        print(f"{rule_name:32s}: {count:10,} rows ({percentage:6.2f}%)")


def print_structured_explanation(raw_rows, cleaned_rows, invalid_rows) -> None:
    """Print the written explanation required for the lab submission."""
    print("\nAct 2 Explanation")
    print("=================")

    print("\n1. Data Validation Process")
    print(
        "The raw taxi data was checked for missing values, non-positive "
        "trip distance, non-positive fare amount, impossible time order "
        "where pickup is after dropoff, non-positive trip duration, and "
        "extreme outliers. These checks identify rows that can distort "
        "fare-per-mile efficiency and zone classification."
    )

    print("\n2. Definition of a Real Trip")
    print(
        "A real trip is defined as a record with complete pickup/dropoff, "
        "distance, and fare values; trip_distance > 0; fare_amount > 0; "
        "dropoff time after pickup time; duration between 0 and 24 hours; "
        "trip_distance <= 100 miles; fare_amount <= 500 dollars; and "
        "efficiency <= 100 dollars per mile."
    )

    print("\n3. Impact of Cleaning")
    print(
        f"The dataset contained {raw_rows:,} rows before cleaning. "
        f"{invalid_rows:,} invalid rows were removed, leaving "
        f"{cleaned_rows:,} real trips for final analysis. After cleaning, "
        "efficiency = fare_amount / trip_distance was recalculated so that "
        "zone labels are based only on valid and realistic trips."
    )

    print("\n4. Why Conclusions Change")
    print(
        "Before cleaning, zero distances, negative fares, reversed times, "
        "and extreme outliers can push trips into incorrect Stable, Stress, "
        "or Unstable zones. After cleaning, the distribution reflects real "
        "taxi behavior more accurately. Therefore, the same formula can "
        "lead to different conclusions when the definition of valid data "
        "changes."
    )

    print("\n5. Strong Conclusion")
    print(
        "Defining what counts as a real trip directly affects how system "
        "behavior is interpreted. If invalid or unrealistic records are "
        "included, the taxi system may appear more unstable, inefficient, "
        "or contradictory than it actually is. A clear real-trip definition "
        "turns raw records into reliable evidence, so the final zone "
        "distribution represents actual system behavior rather than data "
        "quality problems."
    )


def main() -> None:
    try:
        install_missing_packages()

        raw_df = load_data(DATA_FILE)
        prepared_df = prepare_columns(raw_df)
        raw_with_zones = add_efficiency_and_zone(prepared_df)

        invalid_masks = detect_invalid_trips(prepared_df)
        print_invalid_trip_report(prepared_df, invalid_masks)

        cleaned_df = clean_data(prepared_df, invalid_masks)
        cleaned_with_zones = add_efficiency_and_zone(cleaned_df)

        raw_rows = len(prepared_df)
        cleaned_rows = len(cleaned_with_zones)
        invalid_rows = int(invalid_masks["Any invalid rule"].sum())

        print("\nDataset Size")
        print("------------")
        print(f"Before cleaning: {raw_rows:,} rows")
        print(f"After cleaning : {cleaned_rows:,} rows")
        print(f"Rows removed   : {invalid_rows:,} rows")

        before_counts = zone_counts(raw_with_zones)
        after_counts = zone_counts(cleaned_with_zones)

        print("\nZone Distribution Before Cleaning")
        print("---------------------------------")
        print(before_counts)

        print("\nZone Distribution After Cleaning")
        print("--------------------------------")
        print(after_counts)

        plot_zone_comparison(before_counts, after_counts, OUTPUT_PLOT)
        print_structured_explanation(raw_rows, cleaned_rows, invalid_rows)

        print("\nProcessing completed successfully.")
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
