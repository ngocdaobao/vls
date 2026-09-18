import pandas as pd
import csv
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--result_dir", type=str)
    args = arg_parser.parse_args()

    df = pd.read_csv(args.result_dir)

    category = df['Category'].unique()
    print(category)
    success_rate = {}
    for cat in category:
        df_cat = df[df['Category'] == cat]
        num_cat = len(df_cat)
        df_cat_success = len(df_cat[df_cat['Status'] == "success"])
        df_cat_fail = len(df_cat[df_cat['Status'] == "failed"])
        success_rate[cat] = f"{df_cat_success/num_cat*100:.2f}%"

    print("Success Rate by Category:")
    for cat, rate in success_rate.items():
        print(f"{cat}: {rate}")