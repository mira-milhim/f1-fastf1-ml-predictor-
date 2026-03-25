import fastf1
import pandas as pd
import joblib
from pathlib import Path

# Ensure cache exists
Path("data/cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("data/cache")

FEATURES = [
    "quali_position",
    "avg_finish_last3",
    "avg_finish_last5",
    "avg_grid_last3",
    "points_last3",
    "dnf_last5",
    "team_avg_finish_last3",
    "team_points_last3",
    "driver_circuit_avg_finish",
    "team_circuit_avg_finish",
    "round"
]


def main():
    # Load data + model
    hist = pd.read_csv("data/processed/model_data.csv")
    model = joblib.load("models/f1_model.pkl")

    # Ask user
    year = int(input("📅 Enter year: ").strip())
    round_number = int(input("🏎️ Enter round number: ").strip())

    # Load qualifying session safely
    try:
        quali = fastf1.get_session(year, round_number, "Q")
        quali.load()
    except Exception as e:
        print("\n❌ Qualifying data is not available yet for this race.")
        print("⏳ Run this script after qualifying has finished and FastF1 has the data.")
        print(f"🔍 Details: {e}\n")
        return

    # Check results exist
    quali_results = quali.results.copy()
    if quali_results.empty:
        print("\n❌ Qualifying results are empty.")
        print("⏳ Run this script after qualifying data becomes available.\n")
        return

    event_name = quali.event["EventName"]

    rows = []

    for _, q in quali_results.iterrows():
        driver = q["Abbreviation"]
        team = q["TeamName"]

        driver_hist = hist[hist["driver"] == driver].sort_values(["year", "round"])
        team_hist = hist[hist["team"] == team].sort_values(["year", "round"])
        driver_circuit_hist = hist[
            (hist["driver"] == driver) & (hist["event_name"] == event_name)
        ].sort_values(["year", "round"])
        team_circuit_hist = hist[
            (hist["team"] == team) & (hist["event_name"] == event_name)
        ].sort_values(["year", "round"])

        rows.append({
            "driver": driver,
            "team": team,
            "quali_position": q.get("Position"),
            "avg_finish_last3": driver_hist["finish_position"].tail(3).mean(),
            "avg_finish_last5": driver_hist["finish_position"].tail(5).mean(),
            "avg_grid_last3": driver_hist["grid_position"].tail(3).mean(),
            "points_last3": driver_hist["points"].tail(3).sum(),
            "dnf_last5": driver_hist["dnf_flag"].tail(5).sum() if "dnf_flag" in driver_hist.columns else 0,
            "team_avg_finish_last3": team_hist["finish_position"].tail(3).mean(),
            "team_points_last3": team_hist["points"].tail(3).sum(),
            "driver_circuit_avg_finish": driver_circuit_hist["finish_position"].mean(),
            "team_circuit_avg_finish": team_circuit_hist["finish_position"].mean(),
            "round": round_number
        })

    pred_df = pd.DataFrame(rows)

    # Fill missing values safely
    for feature in FEATURES:
        pred_df[feature] = pd.to_numeric(pred_df[feature], errors="coerce")
        fallback = pd.to_numeric(hist[feature], errors="coerce").median()
        if pd.isna(fallback):
            fallback = 0
        pred_df[feature] = pred_df[feature].fillna(fallback)

    # Predict
    pred_df["predicted_finish"] = model.predict(pred_df[FEATURES])
    pred_df = pred_df.sort_values("predicted_finish").reset_index(drop=True)
    pred_df["predicted_rank"] = pred_df.index + 1

    print(f"\n🏁 Predicted Results for {event_name} {year} 🏁\n")

    for _, row in pred_df.iterrows():
        rank = int(row["predicted_rank"])
        driver = row["driver"]
        team = row["team"]
        score = round(row["predicted_finish"], 2)

        if rank == 1:
            emoji = "🥇"
        elif rank == 2:
            emoji = "🥈"
        elif rank == 3:
            emoji = "🥉"
        elif rank <= 10:
            emoji = "🔥"
        else:
            emoji = "📉"

        print(f"{emoji} P{rank:02d} | {driver} ({team}) → {score}")

    print("\n🏆 Expected Podium 🏆\n")
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(pred_df.head(3).iterrows()):
        driver = row["driver"]
        team = row["team"]
        print(f"{medals[i]} {driver} ({team})")

    print("\n⚡ Model Insight: Lower score = better expected finish\n")

    # Save prediction
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    filename = f"data/predictions/{year}_{round_number}_{event_name.replace(' ', '_')}_prediction.csv"
    pred_df.to_csv(filename, index=False)

    print(f"💾 Saved prediction to: {filename}\n")


if __name__ == "__main__":
    main()