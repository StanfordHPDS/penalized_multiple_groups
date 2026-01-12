# import modules
import glob
import pandas as pd

# list all csv files only
csv_files = glob.glob("*.{}".format("csv"))

# append all files together
df_concat = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

df_concat.to_csv("penalty_res.csv")
