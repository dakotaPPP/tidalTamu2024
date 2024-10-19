import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')
df.head()
labels = df["specific.disorder"].unique().tolist()
x = []
y = []
for i, row in df.iterrows():
    y.append(labels.index(row["specific.disorder"]))
    x.append(row.tolist()[8:])

