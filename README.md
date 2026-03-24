🏎️ F1 Race Predictor (FastF1 + Machine Learning)

This project predicts Formula 1 race results using historical race data and qualifying results.

Built using:

FastF1 (data API)
pandas (data processing)
scikit-learn (machine learning)

🚀 Features
📊 Collects official F1 data using FastF1
🧠 Builds ML features (driver form, team performance, circuit history)
🤖 Trains a model to predict finishing positions
🏁 Predicts race results after qualifying
🏆 Outputs podium + full ranking
💾 Saves predictions as CSV

🧩 Project Structure
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
git clone https://github.com/YOUR_USERNAME/f1-race-predictor.git
cd f1-race-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

▶️ How to Use
1. Fetch Data
python src/fetch_data.py
2. Build Dataset
python src/build_dataset.py
3. Train Model
python src/train_model.py
4. Predict Race (after qualifying)
python src/predict_race.py

🔁 Weekly Workflow
After qualifying:
python src/predict_race.py

After race:
python src/fetch_data.py
python src/build_dataset.py
python src/train_model.py

📈 Features Used
Qualifying position
Driver recent performance (last 3–5 races)
Team recent performance
Circuit historical performance
Points & reliability (DNFs)

🧠 Model
GradientBoostingRegressor (scikit-learn)
Predicts finishing position (regression)
Drivers ranked based on predicted values

📊 Example Output
🏁 Predicted Results for Singapore Grand Prix 2025 🏁

🥇 P01 | VER (Red Bull Racing)
🥈 P02 | NOR (McLaren)
🥉 P03 | LEC (Ferrari)
🔥 P04 | RUS (Mercedes)
...