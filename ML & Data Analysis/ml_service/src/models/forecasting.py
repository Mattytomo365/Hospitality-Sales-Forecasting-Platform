import pandas as pd, numpy as np
from src.preprocessing.encoding import apply_onehot_schema, load_onehot_schema
from src.preprocessing.features import add_all_features
from src.dataset.load_save import load_csv

'''
Load trained model & produce forecasts
'''

Lags = (1, 7, 14, 28) # maintain the same day of week
Roll_windows = (7, 14, 28) # weekly windows to get weekly patterns
Warmup = max(max(Lags), max(Roll_windows)) # ensures correct amount of rows in buffer for lags/rolling stats

def forecast_features(model, df, date, features, target_col, internal_events, external_events, weather, holiday, onehot_path="models/onehot_schema.json"):
    '''
    Recreates the same features the model expects, for dates start - end
    Uses history for lags/rolling stats and zeros future onehots unless events are supplied
    '''
    df_hist = df.copy() # dataset of historical data 

    target_date = pd.to_datetime(date)
    last_date = df_hist["date"].max()
    if target_date <= last_date:
        raise ValueError("Target date must be after the last known history date")

    # load saved onehot schema
    schema = load_onehot_schema(onehot_path)

    # create small working buffer
    tail = df_hist.tail(Warmup)[["date", target_col]].copy()

    # walk from day after last_date until target_date
    days = pd.date_range(last_date + pd.Timedelta(days=1), target_date)
    preds = []
    future = pd.DataFrame({"date": days}) # convert to dataframe

    # provide placeholder columns - feature engineering expects these columns in 'future'
    for col in ["weather", "internal_events", "external_events", "holiday"]:
        if col in df_hist.columns and col not in future.columns:
            future[col] = ""

    future[target_col] = np.nan # target placeholder so lag/rolling stats work

    # create a scaffold of history buffer/tail and empty future rows (filled within loop)
    scaffold = pd.concat([tail, future], ignore_index=True)

    # cycle through scaffold - apply onehot schema, add determinstic features, and lags/rolling stats
    for i in range(len(tail), len(scaffold)):

        # set weather with user-inputted data for target date
        if i == len(scaffold) -1 and weather and "weather" and scaffold.columns:
            scaffold.loc[i, "weather"] = weather

        # set internal_events with user-inputted data for target date
        if i == len(scaffold) -1 and internal_events and "internal_events" and scaffold.columns:
            scaffold.loc[i, "internal_events"] = internal_events

        # set external_events with user-inputted data for target date
        if i == len(scaffold) -1 and external_events and "internal_events" and scaffold.columns:
            scaffold.loc[i, "external_events"] = external_events

        # set holiday with user-inputted data for target date
        if i == len(scaffold) -1 and holiday and "holiday" and scaffold.columns:
            scaffold.loc[i, "holiday"] = holiday

        temp = add_all_features(scaffold.iloc[: i+1]) # engineers features on 0-i so past data can be used for lag/roll features
        row = temp.iloc[[-1]] # last row in temp
        row = apply_onehot_schema(row, schema, drop_original=False) # applies saved onehot schema to new row
        row = row.reindex(columns=features, fill_value=0.0) # alinging row to model's expected features

        if row.isna().any(): # guards against NaNs for insufficient buffer history
            raise ValueError("Not enough buffer history for required lags/rolling stats")
        
        # predict current day
        row = row.to_numpy() # features for prediction
        pred = model.predict(row)
        preds.append({"date": scaffold.loc[i, "date"], "forecast": pred})

        # write prediction back into scaffold - used for next lags/rolling stats features
        scaffold.loc[i, target_col] = pred

    # return full series of predictions
    return preds


def forecast(model_id, target_date, target_col, covers, weather, internal_events, external_events, holiday):
    df = load_csv("data/restaurant_data_processed.csv")
    # load model artifact/manifest to get model & associated features!!!
    preds = forecast_features(model, df, target_date, features, target_col, internal_events, external_events, weather, holiday)
    return preds