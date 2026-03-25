# 🏎️ F1 Race Predictor (FastF1 + Machine Learning)

This project predicts Formula 1 race results using historical race data and qualifying results.

Built with:
- FastF1
- pandas
- scikit-learn

## 🚀 Features

- Collects official F1 data using FastF1
- Builds machine learning features from race history
- Trains a model to predict finishing positions
- Predicts race results after qualifying
- Outputs podium + full ranking
- Saves predictions as CSV

## 📊 Features Used

- Qualifying position
- Driver Recent performance
- Team Recent performance
- Circuit history
- Points and reliability

## 🧠 Model

- GradientBoostingRegressor
- Predicts finishing position
- Ranks drivers by predicted score

## ⚙️ Installation
git clone https://github.com/mira-milhim/f1-fastf1-ml-predictor-.git

cd f1-fastf1-ml-predictor-

python -m venv .venv

python -m venv .venv

.venv\Scripts\activate

## ▶️ Usage
Fetch data:

python src/fetch_data.py

Build dataset:

python src/build_dataset.py

Train model:

python src/train_model.py

Predict a race:

python src/predict_race.py

## 🔁 Weekly workflow

Before qualifying:

Race prediction is not available yet in the current version, because the model uses qualifying results as an input.

After qualifying:

python src/predict_race.py

After race:

python src/fetch_data.py

python src/build_dataset.py

python src/train_model.py

## 📁 Project Structure

```text
f1_predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── predictions/
│
├── models/
│   └── f1_model.pkl
│
├── src/
│   ├── fetch_data.py
│   ├── build_dataset.py
│   ├── train_model.py
│   ├── predict_race.py
│   └── weekly_update.py
│
├── requirements.txt
└── README.md
