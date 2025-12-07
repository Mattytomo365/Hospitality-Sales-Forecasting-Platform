from flask import Blueprint, request, jsonify
from .dataset.load_save import load_csv # relative imports   
from .analysis import eda
'''
Flask API analysis endpoints
'''

# Bundle analytics routes in a blueprint
bp_analysis = Blueprint("analytics", __name__)

# monthly averages for chosen metric, used in frontend for user-facing visualisations
@bp_analysis.get("/monthly-avg")
def monthly_avg(): # call twice from angular code to get both sales and covers
    metric = request.args.get("metric", "sales") # reads metric from query param or defaults to "sales"
    df = load_csv("data/df_feature.csv") # change with name of dataset!
    out = eda.monthly_avg(df, metric)
    return jsonify(out.to_dict(orient="records")) # returns list of row dictionaries

# weekday averages for specified month and metric, used in frontend for user-facing visualisations
@bp_analysis.get("/weekday-avg")
def weekday_avg():
    month = int(request.args.get("month"))
    metric = request.args.get("metric", "sales")
    df = load_csv("data/df_feature.csv") 
    out = eda.weekday_avg(df, month, metric)
    return jsonify({"month": month, "label": eda.month_labels.get(month, str(month)), # adds month and label properties on top-level rather than for every column in dataset like monthly_avg
                    "data": out.to_dict(orient="records")})

# percentage uplift for specified month and specified factor and metric
@bp_analysis.get("/uplift")
def uplift():
    factor = request.args.get("factor", "internal_events")
    month = int(request.args.get("month"))
    metric = request.args.get("metric", "sales")
    df = load_csv("data/df_feature.csv")
    out = eda.uplift(df, factor, month, metric)
    return jsonify(out.to_dict(orient="records"))