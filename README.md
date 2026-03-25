# f1-fastf1-ml-predictor-
F1 race prediction model using FastF1, pandas, and machine learning to predict race results based on qualifying and historical performance.

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

⚙️ Installation
git clone https://github.com/mira-milhim/f1-fastf1-ml-predictor-.git
cd f1-fastf1-ml-predictor-

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

▶️ Usage
Fetch data
python src/fetch_data.py
Build dataset
python src/build_dataset.py
Train model
python src/train_model.py
Predict a race
python src/predict_race.py
🔁 Weekly workflow

After qualifying:

python src/predict_race.py

After race:

python src/fetch_data.py
python src/build_dataset.py
python src/train_model.py
📊 Features used
Qualifying position
Driver recent performance
Team recent performance
Circuit history
Points and reliability