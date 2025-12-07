import pandas as pd, hashlib, datetime as dt, numpy as np
import re
'''
Data pre-processing & cleaning
'''

# data normalisation to adhere to naming conventions
def normalise_headers(df):
    out = df.copy() # returning new dataframes avoids input mutation
    out.columns = [col.strip().lower().replace(" ", "_") for col in out.columns]
    return out

# cleanup of string-based columns, keeping values readable and canonical
def standardise_strings(df):
    out = df.copy()
    for col in ["internal_events", "external_events", "holiday"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip().str.lower() # fills null values with space and removes leading/trailing spaces
            s = out[col]
            if col == "holiday" or col == "external_events":
                s = (s.str.replace(r"[|,/]", ";", regex=True) # normalize delimiters to ';' 
                        .str.replace(r"\s*;\s*", ";", regex=True) # trim around ;
                        .str.replace(r";{2,}", ";", regex=True) # collapse ;;
                        .str.strip(";"))
            else:
                s = s.str.replace(r"\s+", " ", regex=True) # collapse internal spaces to single space
    return out

# parsing dates to datetime objects
def parse_dates(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%d/%m/%Y", errors="raise") # converts and raises any exceptions encountered, fails fast
    return out

# ensures necessary columns are numerical in type and erronous data is surfaced
def coerce_numeric(df):
    out = df.copy()
    for col in ["sales", "covers"]:
        s = out[col] # series of values within column
        s = s.where(~s.apply(lambda x: isinstance(x, type)), np.nan) # replaces stray Python type objects with NaN
        s = s.astype(str).str.strip()
        s = s.str.replace("£,", "", regex=True) # removes currency symbols
        s = s.replace({"": np.nan, "None": np.nan, "N/A": np.nan, "-": np.nan}) # treats comman placeholders as NaN
        out[col] = pd.to_numeric(s, errors="coerce") # converts back to numeric, erronous values are nulled
    return out

# handles missing values according to column
def handle_missing(df):
    out = df.copy()
    out["weather"] = (out["weather"].astype("string").str.strip().str.lower().fillna("cloudy")) # fill null weather with 'cloudy' and normalises non-null values

    out["day_of_week"] = out["date"].dt.day_name() # adds day-of-week feature for day-specific apc

    valid = (out["sales"].notna()) & (out["covers"].notna()) & (out["covers"].gt(0)) & (out["covers"].gt(0))
    apc_global = (out.loc[valid, "sales"] / out.loc[valid, "covers"]).median() # median avg £ per cover for fallback use
    apc_dow = ( # median avg £ per cover for each day of the week
        (out.loc[valid, "sales"] / out.loc[valid, "covers"])
        .groupby(out.loc[valid, "day_of_week"])
        .median()
    )

    m_sales = (out["covers"].notna()) & (out["sales"].isna()) # missing sales values with present covers values
    m_cov = (out["covers"].isna()) & (out["sales"].notna()) # missing cover values with present sales values
    m_both = (out["covers"].isna()) & (out["sales"].isna()) # missing sales & covers

    # impute sales (sales = covers * apc)
    if m_sales.any():
        apc = (out.loc[m_sales, "day_of_week"].map(apc_dow)).fillna(apc_global) # maps imputable sales rows through day-specific apc using apc_global as fallback
        out.loc[m_sales, "sales"] = (out.loc[m_sales, "covers"] * apc).round(0)
    
    # impute covers (covers = sales / apc)
    if m_cov.any():
        apc = (out.loc[m_cov, "day_of_week"].map(apc_dow)).fillna(apc_global) # maps imputable covers rows through day-specific apc using apc_global as fallback
        out.loc[m_cov, "covers"] = (out.loc[m_cov, "sales"] / apc).round().clip(lower=0)  

    out = out.loc[~m_both].reset_index(drop=True) # drops rows with no present sales or covers

    summary = { # summry report of imput & drop sums
        "datast cleaned: "
        "imputed_sales_rows": int(m_sales.sum()),
        "imputed_covers_rows": int(m_cov.sum()),
        "dropped_rows_both_missing": int(m_both.sum()),
        "apc_global": float(apc_global),
    }     

    return out.drop(columns=["day_of_week"]), summary

# drop all duplicate dates
def handle_duplicates(df):
    out = df.drop_duplicates(subset=["date"]).copy() # uses subset to treat rows with the same dates as duplicates
    return out

# handles outliers carefully using threshold-style rules (z-score)
def handle_outliers(df, z=5.0):
    out = df.copy()
    keep = out["holiday"] != ""
    for col in ["sales", "covers"]:
        mu, sd = out[col].mean(), out[col].std(ddof=0)
        within = (out[col] - mu).abs() <= z * sd # reveals values with deviations from the mean outside the threshold
        out = out[within | keep].copy() # ignores rows during holiday periods
    return out.reset_index(drop=True)

# centralises cleaning & pre-processing, imported into clean_data.py script
def clean_data(df):
    df = (df
            .pipe(normalise_headers)
            .pipe(standardise_strings)
            .pipe(parse_dates)
            .pipe(coerce_numeric))
    
    df, report = handle_missing(df)

    df = (df
            .pipe(handle_duplicates)
            .pipe(handle_outliers))
    
    return df, report
          
