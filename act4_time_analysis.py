from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("yellow_tripdata_2023-01.parquet")
HOURLY_PLOT = Path("act4_hourly.png")
DAILY_PLOT = Path("act4_daily.png")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load pickup datetime from the NYC Yellow Taxi parquet file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_parquet(file_path, columns=["tpep_pickup_datetime"])


def prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, and week from pickup datetime."""
    prepared = df.copy()
    prepared["tpep_pickup_datetime"] = pd.to_datetime(
        prepared["tpep_pickup_datetime"], errors="coerce"
    )
    prepared = prepared.dropna(subset=["tpep_pickup_datetime"])

    prepared["hour"] = prepared["tpep_pickup_datetime"].dt.hour
    prepared["day"] = prepared["tpep_pickup_datetime"].dt.date
    prepared["week"] = prepared["tpep_pickup_datetime"].dt.isocalendar().week
    prepared["pickup_hour"] = prepared["tpep_pickup_datetime"].dt.floor("h")
    prepared["pickup_day"] = prepared["tpep_pickup_datetime"].dt.floor("D")

    return prepared


def aggregate_trips(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Aggregate trips per hour and per day."""
    hourly_trips = df.groupby("pickup_hour").size().sort_index()
    daily_trips = df.groupby("pickup_day").size().sort_index()
    return hourly_trips, daily_trips


def plot_hourly_trend(hourly_trips: pd.Series, output_path: Path) -> None:
    """Save the hourly trip-count signal as a line plot."""
    plt.figure(figsize=(12, 5))
    plt.plot(hourly_trips.index, hourly_trips.values, color="steelblue", linewidth=1)
    plt.title("NYC Yellow Taxi Trips per Hour")
    plt.xlabel("Pickup Hour")
    plt.ylabel("Number of Trips")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_daily_trend(daily_trips: pd.Series, output_path: Path) -> None:
    """Save the daily trip-count signal as a line plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        daily_trips.index,
        daily_trips.values,
        color="seagreen",
        marker="o",
        linewidth=2,
    )
    plt.title("NYC Yellow Taxi Trips per Day")
    plt.xlabel("Pickup Day")
    plt.ylabel("Number of Trips")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_peaks_and_drops(signal: pd.Series, top_n: int = 5) -> tuple[pd.Series, pd.Series]:
    """Return the largest peaks and drops in a time signal."""
    peaks = signal.sort_values(ascending=False).head(top_n)
    drops = signal.sort_values(ascending=True).head(top_n)
    return peaks, drops


def detect_irregular_spikes(signal: pd.Series) -> pd.Series:
    """Detect unusually high points using a simple mean plus two-std rule."""
    threshold = signal.mean() + (2 * signal.std())
    return signal[signal > threshold].sort_values(ascending=False)


def print_signal_summary(hourly_trips: pd.Series, daily_trips: pd.Series) -> None:
    """Print time-signal analysis and interpretation."""
    hourly_peaks, hourly_drops = find_peaks_and_drops(hourly_trips)
    daily_peaks, daily_drops = find_peaks_and_drops(daily_trips)
    hourly_spikes = detect_irregular_spikes(hourly_trips)
    daily_spikes = detect_irregular_spikes(daily_trips)

    hourly_by_clock = hourly_trips.groupby(hourly_trips.index.hour).mean()
    busiest_clock_hours = hourly_by_clock.sort_values(ascending=False).head(5)
    quietest_clock_hours = hourly_by_clock.sort_values(ascending=True).head(5)

    print("Act 4: Time-Based Signal Analysis")
    print("=================================")

    print("\nAggregated Dataset")
    print("------------------")
    print(f"Hourly points: {len(hourly_trips):,}")
    print(f"Daily points : {len(daily_trips):,}")
    print(f"Total trips  : {int(hourly_trips.sum()):,}")

    print("\nHourly Peaks")
    print("------------")
    for timestamp, count in hourly_peaks.items():
        print(f"{timestamp}: {count:,} trips")

    print("\nHourly Drops")
    print("------------")
    for timestamp, count in hourly_drops.items():
        print(f"{timestamp}: {count:,} trips")

    print("\nDaily Peaks")
    print("-----------")
    for timestamp, count in daily_peaks.items():
        print(f"{timestamp.date()}: {count:,} trips")

    print("\nDaily Drops")
    print("-----------")
    for timestamp, count in daily_drops.items():
        print(f"{timestamp.date()}: {count:,} trips")

    print("\nRepeating Daily Cycle")
    print("---------------------")
    print("Busiest average clock hours:")
    for hour, count in busiest_clock_hours.items():
        print(f"{hour:02d}:00 - average {count:,.0f} trips")
    print("Quietest average clock hours:")
    for hour, count in quietest_clock_hours.items():
        print(f"{hour:02d}:00 - average {count:,.0f} trips")

    print("\nIrregular Spikes")
    print("----------------")
    if hourly_spikes.empty and daily_spikes.empty:
        print("No strong irregular spikes were detected using the two-std rule.")
    else:
        if not hourly_spikes.empty:
            print("Hourly spikes:")
            for timestamp, count in hourly_spikes.head(10).items():
                print(f"{timestamp}: {count:,} trips")
        if not daily_spikes.empty:
            print("Daily spikes:")
            for timestamp, count in daily_spikes.head(10).items():
                print(f"{timestamp.date()}: {count:,} trips")

    print("\nInterpretation")
    print("--------------")
    print(
        "The hourly signal preserves short-term variation, including rush-hour "
        "cycles, late-night drops, and isolated spikes. These details show how "
        "taxi demand changes within a day."
    )
    print(
        "The daily signal smooths out hour-level noise. Repeating within-day "
        "peaks and drops disappear under daily aggregation, while broader "
        "changes across dates remain visible."
    )
    print(
        "Patterns that repeat at similar clock hours are persistent daily "
        "cycles. Single unusually high or low points that do not repeat are "
        "better interpreted as noise or one-time events rather than stable "
        "system behavior."
    )


def main() -> None:
    raw_df = load_dataset(DATA_FILE)
    time_df = prepare_time_features(raw_df)
    hourly_trips, daily_trips = aggregate_trips(time_df)

    plot_hourly_trend(hourly_trips, HOURLY_PLOT)
    plot_daily_trend(daily_trips, DAILY_PLOT)
    print_signal_summary(hourly_trips, daily_trips)

    print("\nSaved Plots")
    print("-----------")
    print(f"Hourly trend: {HOURLY_PLOT}")
    print(f"Daily trend : {DAILY_PLOT}")


if __name__ == "__main__":
    main()
