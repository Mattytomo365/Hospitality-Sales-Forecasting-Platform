from flask import Flask, Blueprint, request, jsonify
from models.registry import list_models
from dataset.load_save import load_csv
from models.forecasting import forecast
from pathlib import Path
import pandas as pd
import json, time, joblib

'''
Flask API core endpoints
'''

bp_core = Blueprint("core", __name__)

@bp_core.get("/models")
def models_list():
    return jsonify(list_models())

@bp_core.post("/forecast")
def forecast():
    # retrive parameters from request
    model_id = request.args.get("model_id")
    target_date = request.args.get("target_date")
    covers = 
    weather = 
    internal_events = 
    external_events = 
    holiday = 
    preds = forecast(model_id, target_date, "sales", covers, weather, internal_events, external_events, holiday)

    if not model_id or pd.isna(target_date):
        return jsonify({"error": "invalid parameters"}), 400
    
    # load and activate model
    model_dir = Path("models") / model_id
    artifact = joblib.load(model_dir / "model.joblib") # reads model artifact
    model = artifact["estimator"]
    features = artifact["features"]

    # load cleaned, engineered, processed dataset
    df = load_csv("data/restaurant_data_processed.csv")

    # call the forecasting method to obtain result
    result = forecast(model, df, target_date, features, target_col="sales")
    return result
