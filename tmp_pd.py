import pandas as pd
import urllib.request
url='https://zenodo.org/records/10952917/files/IRENASTAT_capacities_2000-2023.csv'
with urllib.request.urlopen(url) as resp:
    df=pd.read_csv(resp, comment='#', quotechar='"')
print(df.shape)
print(df.head())
