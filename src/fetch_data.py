import fastf1
import pandas as pd
from pathlib import Path

Path("data/cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("data/cache")


def fetch_year_data(year):
    records = []

    schedule = fastf1.get_event_schedule(year)

    # make "now" timezone-aware in UTC
    now = pd.Timestamp.now(tz="UTC")

    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        event_name = event["EventName"]

        if round_number == 0:
            continue

        try:
            race_date = pd.to_datetime(event["Session5DateUtc"], errors="coerce")

            if pd.isna(race_date):
                print(f"Skipped {year} round {round_number} ({event_name}): no race date")
                continue

            # make sure race_date is timezone-aware UTC too
            if race_date.tzinfo is None:
                race_date = race_date.tz_localize("UTC")
            else:
                race_date = race_date.tz_convert("UTC")

            if race_date > now:
                print(f"Skipped {year} round {round_number} ({event_name}): future race")
                continue

            race = fastf1.get_session(year, round_number, "R")
            race.load()

            quali = fastf1.get_session(year, round_number, "Q")
            quali.load()

            race_results = race.results.copy()
            quali_results = quali.results.copy()

            if race_results.empty or quali_results.empty:
                print(f"Skipped {year} round {round_number} ({event_name}): empty results")
                continue

            quali_map = quali_results.set_index("Abbreviation")["Position"].to_dict()

            for _, row in race_results.iterrows():
                driver = row["Abbreviation"]

                records.append({
                    "year": year,
                    "round": round_number,
                    "event_name": event_name,
                    "driver": driver,
                    "team": row["TeamName"],
                    "grid_position": row.get("GridPosition"),
                    "quali_position": quali_map.get(driver),
                    "finish_position": row.get("Position"),
                    "points": row.get("Points"),
                    "status": row.get("Status"),
                })

            print(f"Loaded {year} round {round_number} ({event_name})")

        except Exception as e:
            print(f"Skipped {year} round {round_number} ({event_name}): {e}")

    return pd.DataFrame(records)


def main():
    all_data = []

    for year in [2023, 2024, 2025, 2026]:
        df = fetch_year_data(year)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("No data fetched.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    final_df.to_csv("data/raw/results.csv", index=False)

    print(final_df.head())
    print(f"Saved {len(final_df)} rows to data/raw/results.csv")


if __name__ == "__main__":
    main()