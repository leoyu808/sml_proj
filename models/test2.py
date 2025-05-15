import pandas as pd
import pickle

with open("df.pkl", "rb") as f:
    df = pickle.load(f)
    df[0] = df[0].drop(columns=['MATCH_ID'], errors='ignore')
    print(df[0]["MATCH_ID"])