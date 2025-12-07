import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
'''
Model performance evaluation using regression metrics and visualisations
'''

def calculate_metrics(Y_test, pred, kind):
    # mean absolute error
    mae = float(mean_absolute_error(Y_test, pred))
    # root mean squared error
    rmse = float(root_mean_squared_error(Y_test, pred, squared=False))
    # R2 score
    r2 = float(r2_score(Y_test, pred))
    # mean absolute percentage error
    mask = (np.abs(Y_test) > 0) # excludes 0's to avoid division by 0
    mape = float((np.abs(Y_test[mask] - pred) / Y_test[mask])).mean() * 100

    metrics = {"kind": kind, "test_days": 56, "MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}
    return metrics

