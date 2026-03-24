import pandas as pd
from pathlib import Path


def add_dnf_flag(df):
    df = df.copy()
    df["status"] = df["status"].astype(str)
    df["dnf_flag"] = df["status"].str.contains(
        "Accident|Collision|Retired|DNF|Disqualified",
        case=False,
        na=False
    ).astype(int)
    return df


def add_driver_features(df):
    df = df.sort_values(["driver", "year", "round"]).copy()

    df["avg_finish_last3"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["avg_finish_last5"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df["avg_grid_last3"] = (
        df.groupby("driver")["grid_position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["points_last3"] = (
        df.groupby("driver")["points"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )

    df["dnf_last5"] = (
        df.groupby("driver")["dnf_flag"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )

    return df


def add_team_features(df):
    df = df.sort_values(["team", "year", "round"]).copy()

    df["team_avg_finish_last3"] = (
        df.groupby("team")["finish_position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["team_points_last3"] = (
        df.groupby("team")["points"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )

    return df


def add_circuit_features(df):
    df = df.sort_values(["event_name", "year", "round"]).copy()

    df["driver_circuit_avg_finish"] = (
        df.groupby(["driver", "event_name"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    df["team_circuit_avg_finish"] = (
        df.groupby(["team", "event_name"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    return df


def main():
    df = pd.read_csv("data/raw/results.csv")

    for col in ["grid_position", "quali_position", "finish_position", "points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["finish_position"]).copy()

    df = add_dnf_flag(df)
    df = add_driver_features(df)
    df = add_team_features(df)
    df = add_circuit_features(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/model_data.csv", index=False)

    print(df.head())
    print(f"Saved processed dataset with {len(df)} rows")


if __name__ == "__main__":
    main()