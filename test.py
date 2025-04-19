from scipy.io import arff
import pandas as pd
import io

with open('diabetes.arff', 'rb') as f:
    data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object and not df[col].empty and isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode('utf-8')

    print(df.head())
