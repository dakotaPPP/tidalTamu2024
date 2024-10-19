import pandas as pd
df = pd.read_csv('data.csv')
df.head()
labels = df["specific.disorder"].unique()


