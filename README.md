# Gold Price Prediction — ML Project
A machine learning project to predict whether gold prices will go **UP or DOWN** the next day, using historical daily gold price data.
---
## What This Project Does

Instead of predicting the exact gold price, this project predicts the **direction** of the price movement — a classification problem. This is more useful in practice because it gives a clear signal.

- **Input:** Historical gold price data (Open, High, Low, Close, Volume)
- **Output:** 1 = Price will go UP tomorrow, 0 = Price will go DOWN
- **Final Model:** XGBoost Classifier
- **Final Accuracy:** 57% (random guessing = 50%)
---

## Project Structure

```
gold-price-prediction/
│
├── data/
│   └── gold_historical_data.csv      # Raw dataset
│
├── gold_price_eda.ipynb              # Main notebook (has all milestones in it)
├── README.md                         # Readme file 
```

---

## Milestones

### Milestone 1 — EDA
* Loaded and inspected dataset (2,510 rows, 7 columns, zero missing values)
* Converted Date column to datetime, extracted Year/Month/Day/Weekday
* Created 5 visualizations: price over time, distribution, correlation heatmap, boxplot by year, volume vs price scatter

### Milestone 2 — Data Preparation
* No missing values or categorical columns to handle
* Applied MinMaxScaler for feature scaling
* Split data: 70% train / 24% validation / 6% test

### Milestone 3 — Baseline Model
* Trained Linear Regression to predict gold closing price
* Baseline results: RMSE = 7.68, MAE = 4.45
* Plotted Actual vs Predicted to visually verify

### Milestone 4 — XGBoost Model
* Switched from regression to classification (UP/DOWN direction)
* Built lag features using `Prev_Close` to prevent data leakage
* Features: Return_1d, Return_5d, Vol_5d, Vol_Ratio, Z_Score_20, RSI
* Used **chronological split** (80% train / 20% test) — critical for time series
* Used **RobustScaler** (handles financial outliers better)
* Handled class imbalance with `scale_pos_weight`
* Tuned hyperparameters with `RandomizedSearchCV` + `TimeSeriesSplit`
* **Final accuracy: 57%**

### Milestone 5 — Model Finalization
* Compared XGBoost vs Random Forest
* XGBoost selected as final model (more balanced, higher accuracy)
* Documented strengths and limitations

---

## Results

| Metric | Price Down | Price Up |
-----------------------------------
| Precision | 0.46 | 0.62 |
| Recall | 0.37 | 0.71 |
| F1 Score | 0.41 | 0.66 |
| **Overall Accuracy** — **0.57** |

**Model Comparison:**

| Model | Accuracy | DOWN Recall | UP Recall |
---------------------------------------------
| XGBoost (Final) | **0.57** | 0.37 | 0.71 |
| Random Forest | 0.53 | 0.08 | 0.95 |

XGBoost was chosen because it was more balanced — Random Forest almost completely ignored DOWN days.

---

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/NainaKalra/gold-price-prediction
cd gold-price-prediction
```
2. Install:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```
3. Open the notebook:
```bash
jupyter notebook gold_price_eda.ipynb
```
4. Run all cells from top to bottom.
---

## Limitations

- The model is better at predicting UP days (71% recall) than DOWN days (37% recall).
- It only uses price-based technical indicators, as right now there are no external data like USD index or inflation.
- Since it was a small dataset, there were 2,510 rows of daily data (about 10 years).
- Decision threshold is 0.35 instead of the standard 0.5, showing low model confidence.
---

## Future Improvements

- I will add external features like USD index, inflation rate, global news sentiment.
- Trying LSTM or other deep learning models for sequence prediction will probably help as well.

