# F1 Sprint Race Prediction

Predicting where drivers finish in Formula 1 **sprint races** using machine learning.

Most current F1 prediction work focuses on Sunday Grand Prix results. Sprint races, which were added in 2021, does not have much analysis. They are a shorter distance, have no pit stops, and have less data. This project looks at how Grand Prix prediction methods translate over to Sprints, and how to get high accuracy with the smaller sprint datasets.

**Best result:** a Random Forest regressor that places drivers within **~2.3 positions** of their actual sprint finish on average (**R² = 0.71**). That is a **21% lower error** than a baseline trained only on raw sprint data.

This was a team project for DS 675 (Machine Learning) at NJIT. I built the sprint prediction model; my teammates covered Grand Prix and DNF prediction. The full team report is in [`Formula_1_Predictions.pdf`](Formula_1_Predictions.pdf).

## Data
 
The data is the [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) dataset on Kaggle. It uses `sprint_results.csv`, `results.csv`, `races.csv` and the constructor tables.
 
Sprint data only goes back to 2021. At the time of the project, this gave data for **18 sprint weekends and 360 driver-race entries**, too few to train a strong model on sprint data alone. The main challenge was finding a way to use the much larger Grand Prix history without muddying the sprint predictions.

## Approach

### Feature Engineering

Grand Prix and sprint results are combined into one timeline, with a flag marking which is which. Each driver-race row then gets features computed only from races *before* it, so no future information leaks in:
 
| Feature | What it captures |
|---|---|
| Historical average position | The driver's long-run baseline |
| Recent form (rolling average) | Current momentum, including car upgrades and slumps |
| Circuit-specific average | How the driver has done at this particular track |
| Cumulative position differential | How often the driver gains or loses places from the starting grid |
| Cumulative podiums | Number of top-3 finishes so far |
| Grid position | Where the driver starts |
 
Exploratory analysis also checked how much results depend on the car and how much on the driver. Teammates' finishing positions correlate at **r = 0.51**, so the car matters a lot, but it doesn't explain everything.
 
### Models and training setups
 
Two types of models, **Random Forest** and **XGBoost**, were tested with three different training setups:
 
1. **Sprint only:** raw sprint data, sprint-only features
2. **Combined features, sprint only:** features built from the full GP + sprint history, trained and tested on sprint rows
3. **Combined features, train on all:** trained on GP and sprint rows, tested on sprint rows only
The target is finishing position, treated as a regression problem. Predicting 2nd for a driver who finished 5th should count as a smaller mistake than predicting 20th, and classification would score both as simply wrong.
 
## Results
 
Mean absolute error (MAE) on the sprint test set, measured in finishing positions (lower is better):
 
| Setup | Random Forest | XGBoost |
|---|---|---|
| 1. Sprint only | 2.87 | 2.72 |
| **2. Combined features, sprint only** | **2.28** | 2.62 |
| 3. Combined features, train on all | 2.87 | 2.49 |
 
The best model is Random Forest with setup 2: **MAE 2.28, RMSE 2.97, R² 0.71.**
 
**Takeaways:**
 
- **Grand Prix history helps most as features, not as training rows.** Setup 2 works best: the model learns from sprint results but uses what the driver has done in every race. Training directly on Grand Prix rows (setup 3) brings in a different kind of race and doesn't help the Random Forest.
- **Grid position is the strongest predictor, followed by recent form.** That makes sense for a short race with no pit stops, where there are few chances to gain places.
- **RMSE is higher than MAE (2.97 vs 2.28) because of a few large misses.** These mostly come from DNFs and first-lap incidents, which none of the features can predict.
- **The model never predicted 1st or 20th.** Random Forest averages across many trees, which pulls predictions toward the middle of the field.
I also tried predicting podium finishes as a yes/no classification, but predicting the exact position worked better.
 
## Running it
 
The notebook was written on Kaggle, and the data paths point to Kaggle's `/kaggle/input/` directory.
 
**On Kaggle (easiest):** create a new notebook, attach the *Formula 1 World Championship (1950–2024)* dataset, upload `sprint-prediction.ipynb` and run all cells.
 
**Locally:**
 
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn statsmodels jupyter
```
 
Download the dataset from Kaggle, then update the `pd.read_csv(...)` paths in the "Loading the data" cell to point at your local folder.
 
## Tech
 
Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · matplotlib · seaborn
 
## Team
 
Kunal Thakker, Patrick DeMarinis, Bartosz Protasewicz
