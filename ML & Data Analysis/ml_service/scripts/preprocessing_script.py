import pandas as pd
from src.dataset.load_save import load_csv, save_csv # absolute imports
from src.preprocessing.cleaning import clean_data
from src.preprocessing.encoding import fit_onehot_schema, save_onehot_schema, load_onehot_schema, apply_onehot_schema
from src.preprocessing.features import add_all_features

def main():

    # data cleaning
    df = load_csv('./data/restaurant_data.csv')
    print(df)
    df_clean, report = clean_data(df)
    print(df_clean)
    print(report)
    print

    # data encoding
    schema = fit_onehot_schema(df_clean)
    save_onehot_schema(schema, "models/onehot_schema.json")

    df_onehot = apply_onehot_schema(df_clean, schema, drop_original=False)
    print(df_onehot)

    # feature engineering
    df_feature = add_all_features(df_onehot)
    print(df_feature)
    df_feature.to_csv("data/restaurant_data_processed.csv", index=False, mode="w",date_format="%Y-%m-%d")
    

if __name__ == "__main__": # used for running script outside of vscode, add argparsing to complete configuration
    main()