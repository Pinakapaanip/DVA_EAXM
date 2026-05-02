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
    """Load only the columns required from the parquet file."""
    import pandas as pd

    required_columns = [
        "tpep_pickup_datetime",
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

    print(f"Loaded {len(df):,} rows.")
    return df


def process_data(df):
    """Clean data and create efficiency and zone columns."""
    print("Cleaning data...")
    original_rows = len(df)

    df = df.dropna()
    df = df[df["trip_distance"] != 0].copy()

    removed_rows = original_rows - len(df)
    print(f"Removed {removed_rows:,} rows with nulls or zero trip distance.")

    if df.empty:
        raise ValueError("No data remains after cleaning.")

    print("Creating efficiency and zone columns...")
    df["efficiency"] = df["fare_amount"] / df["trip_distance"]

    df["zone"] = "Unstable"
    df.loc[df["efficiency"] < 7, "zone"] = "Stable"
    df.loc[df["efficiency"].between(7, 9, inclusive="both"), "zone"] = "Stress"

    return df


def print_results(df) -> None:
    """Print first rows and count of each zone."""
    print("\nFirst 5 rows of processed data:")
    print(df.head())

    print("\nCount of each zone:")
    zone_counts = df["zone"].value_counts().reindex(
        ["Stable", "Stress", "Unstable"], fill_value=0
    )
    print(zone_counts)


def plot_zone_counts(df, output_path: Path) -> None:
    """Plot and save a bar chart showing zone counts."""
    import matplotlib.pyplot as plt

    print(f"Creating bar chart and saving it as {output_path}...")

    zone_counts = df["zone"].value_counts().reindex(
        ["Stable", "Stress", "Unstable"], fill_value=0
    )

    plt.figure(figsize=(8, 5))
    zone_counts.plot(kind="bar", color=["green", "orange", "red"])
    plt.title("Count of Trips by Zone")
    plt.xlabel("Zone")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    try:
        plt.show(block=False)
        plt.pause(2)
    except Exception as error:
        print(f"Could not display plot window: {error}")
    finally:
        plt.close()

    print(f"Plot saved successfully: {output_path}")


def main() -> None:
    try:
        install_missing_packages()
        df = load_data(DATA_FILE)
        processed_df = process_data(df)
        print_results(processed_df)
        plot_zone_counts(processed_df, OUTPUT_PLOT)
        print("\nProcessing completed successfully.")
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
