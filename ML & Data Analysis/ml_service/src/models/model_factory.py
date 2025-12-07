from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
import numpy as np, pandas as pd
'''
A factory returning the right estimator for a specified ML algorithm
'''
exclude = {"date","sales","covers","weather","internal_events","external_events","holiday"}

# chooses appropriate model input features
def feature_cols(df):
    return [c for c in df.columns if c not in exclude]

# calculates mean absolute percentage error metric for fold parameter combinations???
def mape(y, pred):
     return float((np.abs(y - pred) / y)).mean() * 100

def make_estimator(kind, params):
    # Random Forest
    if kind == "rf":
         return RandomForestRegressor(random_state=42, n_jobs=-1, **params) # enough trees to stabalise without increasing process time (n_estimators), uses all cores (n_job)
    # XGBoost
    if kind == "xgb":
         return XGBRegressor(random_state=42, n_jobs=-1, **params)
    # pipelines used with linear models to fit StandardScaler (scales data) before fitting the estimator
    # Ridge
    if kind == "ridge":
         return make_pipeline(StandardScaler(), Ridge(random_state=42, **params)) # sets higher penality strength (alpha) for more bias and less variance
    raise ValueError(f"Unkown kind: {kind}")

# splits training data using a moving train/test set
def rolling_splits(dates, n_folds=3, valid_days=28):
     end = dates.max()
     for k in range(n_folds, 0, -1): # forms train/validation windows
          test_end = end - pd.Timedelta(days=(k-1) * valid_days) # prevents overlapping folds
          test_start = test_end - pd.Timedelta(days=valid_days - 1) # start of kth validation window
          train_end = test_start - pd.Timedelta(days=1)
          train_mask = dates <= train_end # selects training portion of fold
          test_mask = dates.between(test_start, test_end) # selects testing portion of fold
          if train_mask.shape[0] == 0 or test_mask.shape[0] == 0: # train/test validation
               raise ValueError("Train or test split has zero rows")
          yield train_mask, test_mask # returns each fold until loop ends

# determine optimal hyper-parameter combinations using folds returned from rolling_splits
def grid_search(train, features, target, kind, param_grid):
     dates = train["date"]
     X = train[features].to_numpy # feature matrix
     y = train[target].to_numpy # target vector
     best_score, best_params = np.inf, None # keeps track of lowest average validation error and its parameters

     for params in ParameterGrid(param_grid): # iterates over each hyper-parameter combination manually
          fold_scores = []
          for train_mask, test_mask in rolling_splits(dates):
               model = make_estimator(params)
               model.fit(X[train_mask], y[train_mask]) # training data within training window of fold
               prediction = model.predict(X[test_mask]) # predict on validation window for fold
               fold_scores.append(mape(y[test_mask], prediction)) # computes MAPE for current fold
          avg = float(np.mean(fold_scores)) # uses average validation error to pick best parameters
          if avg < best_score:
               best_score, best_params = avg, params
     return {"score": best_score, "params": best_params}