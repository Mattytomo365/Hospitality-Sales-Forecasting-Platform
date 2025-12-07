# scripts/train.py
from src.models.model_factory import make_estimator, grid_search, feature_cols
from src.models.training import time_split, train_model, save_model
from src.models.evaluation import calculate_metrics
from src.dataset.load_save import load_csv


def main(data_path="data/restaurant_data_processed.csv", target="sales", param_grid=None, test_days=56, n_folds=3, valid_days=28, activate=True):
    df = load_csv(data_path)
    features = feature_cols(df)
    train, test = time_split(df)

    # define param grids for each model type
    Grids = {
        "ridge": { # Ridge
            "ridge__alpha": [0.1, 0.3, 1.0, 3.0, 10.0], # pipeline returns StandardScaler
        },

        "rf": { # Random Forest
            "n_estimators": [400, 700],
            "max_depth": [None, 12],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.8],
        },

        "xgb": { # XGBoost
            "n_estimators": [600, 900],   # you can also use early stopping instead of sweeping this widely
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_lambda": [1.0, 3.0, 10.0],
        },
    }

    for kind, grid in Grids.items():
        # tune on train only
        best = grid_search(train, features, target, kind, param_grid=grid)
        print("Best CV: ", best)

        # refit on train with best hyper-parameters
        model = make_estimator(kind, best["params"])
        y_test, preds = train_model(train, test, kind, param_grid, features, target)
        metrics = calculate_metrics(y_test, preds, kind) # evaluates tuned model performance
        print(f"Model Performance ({kind})", metrics)

        # save artifacts + manifest
        save_model(model, target, features, best["params"], metrics)
        print("Model saved")



if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    main()