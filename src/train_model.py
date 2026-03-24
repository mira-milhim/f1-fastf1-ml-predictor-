import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

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
    df = pd.read_csv("data/processed/model_data.csv")

    if df.empty:
        raise ValueError("model_data.csv is empty. Run fetch_data.py and build_dataset.py first.")

    df = df.dropna(subset=["finish_position"]).copy()

    print("Rows per year:")
    print(df["year"].value_counts().sort_index())

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        median_value = df[feature].median()
        if pd.isna(median_value):
            median_value = 0
        df[feature] = df[feature].fillna(median_value)

    years = sorted(df["year"].dropna().unique())

    if len(years) < 2:
        raise ValueError(
            f"Need at least 2 years of data to train/test split. Found only: {years}"
        )

    latest_year = max(years)

    train_df = df[df["year"] < latest_year].copy()
    test_df = df[df["year"] == latest_year].copy()

    if train_df.empty:
        raise ValueError(
            "Training set is empty. You probably fetched only the latest year."
        )

    if test_df.empty:
        raise ValueError(
            "Test set is empty. Check your dataset years."
        )

    X_train = train_df[FEATURES]
    y_train = train_df["finish_position"]

    X_test = test_df[FEATURES]
    y_test = test_df["finish_position"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("MAE:", round(mae, 3))

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/f1_model.pkl")
    print("Model saved to models/f1_model.pkl")


if __name__ == "__main__":
    main()