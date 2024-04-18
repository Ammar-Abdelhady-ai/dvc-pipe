import os
import pandas as pd


out_path = os.path.join(os.getcwd(), "outs")
os.makedirs(out_path, exist_ok=True)

cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, "dataset.csv"))

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df.drop(index=df[df["Age"] > 80].index.to_list(), axis=0, inplace=True)

df.to_csv(os.path.join(out_path, "prepared_df.csv"), index=False)