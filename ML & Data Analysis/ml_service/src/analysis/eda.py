import os, numpy as np, pandas as pd, datetime as dt
import matplotlib.pyplot as plt
'''
User-facing eda on historical data to uncover trends and patterns
'''

dow_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
month_labels = {i: pd.Timestamp(2024, i, 1).strftime("%b") for i in range(1,13)} # month labels based off dates within dataset

# mean metric per calendar month
def monthly_avg(df, metric):
    d = df["date"]
    month = d.dt.month # returns 1-12
    month_m = (df.groupby(month)[metric].agg(value="mean", n_days="count") # aggregating groups to compute mean and n_days 
               .reindex(index=range(1, 13)).fillna({"value": 0, "n_days": 0}) # maintains clean fixed order with no missing months
               .rename_axis("month") # renames index to 'month'
               .reset_index()) # brings the renamed index into a new column

    # additional columns for user interface refinement
    month_m["label"] = month_m["month"].map(month_labels) # converting numerical month names to strings
    month_m["value"] = month_m["value"].astype(float).round(2) # rounding
    month_m["n_days"] = month_m["n_days"].astype(int)
    return month_m[["month", "label", "value", "n_days"]]

# mean metric by weekday for a specified month
def weekday_avg(df, month, metric):
    d = df["date"]
    m = d.dt.month.eq(int(month)) # locates month within dataset

    if not m.any():  # fallback
        return pd.DataFrame({"dow": dow_order, "value": [0]*7}) # => [0, 0, 0, 0, 0, 0, 0]
    
    dow = d.dt.day_name().str[:3] # returns "Mon", "Tue" etc
    day_m = (df.loc[m].groupby(dow)[metric].agg(value="mean") # calculates weekday-based average for metric and names column 'value'
            .reindex(dow_order).fillna(0.0) # maintains fixed order with no missing days
            .rename_axis("dow") # renames index to 'dow'
            .reset_index()) 
    
    day_m["value"] = day_m["value"].astype(float).round(2) # rounding for user interface
    return day_m[["dow", "value"]]


# compute uplift of specified factors against specified metric
def uplift(df, factor, month, metric, sep=";"):
    d = df["date"]
    m = d.dt.month.eq(int(month))

    if not m.any(): # fallback
        return pd.DataFrame(columns=["tag","n","avg","baseline","uplift"])
    
    sub = df.loc[m].copy()
    s = sub[factor].fillna("").astype(str).str.strip().str.lower() # forces robust strings

    tags_list = s.apply(lambda val: [tag.strip() for tag in val.split(sep) if tag.strip() and tag.strip() != "none"])
    base_mask = tags_list.str.len().eq(0) # baseline = days without events
    baseline = (sub.loc[base_mask])[metric].mean() # mean metric of days without events
    baseline_weather = float(df[metric].mean()) # baseline for entire metric

    # add helper columns 
    sub["tags"] = tags_list # attach tag if any

    # explode tags to individual rows
    sub = (sub.explode("tags").rename(columns={"tags": "tag"}))
    sub = sub[["date", "tag", metric]].drop_duplicates(subset=["date", "tag"])

    if factor == "weather":
        sub["baseline"] = baseline_weather

        # filter out rows with an invalid baseline
        sub = sub.loc[sub["baseline"].notna()]

        # fallback
        if sub.empty:
            return pd.DataFrame(columns=["tag","n","avg","baseline","uplift"])

    else:
        sub["baseline"] = baseline

        # filter out rows with no event or an invalid baseline
        sub = sub.loc[~base_mask & sub["baseline"].notna()]

        # fallback
        if sub.empty:
            return pd.DataFrame(columns=["tag","n","avg","baseline","uplift"])
        
    # calculate percentage uplift per row against baseline
    sub["uplift_row"] = 100.0 * (sub[metric] - sub["baseline"]) / sub["baseline"]

    # aggregate per tag - collapse daily rows into a few group-level numbers
    tab = (sub.groupby("tag")
        .agg(n_days=("date", "nunique") # unique days tag occurs
             , avg=(metric, "mean") # mean metric on tagged days
             , uplift_tag=("uplift_row", "mean")) # mean % uplift across occurrences
        .reset_index().sort_values("avg", ascending=False))
    
    return tab
    
        
            



    

