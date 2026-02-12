import pandas as pd

def clean_population_data(path):
    df = pd.read_csv(path)

    # Remove missing values
    df = df.dropna()

    # Convert numeric columns
    df["Population"] = pd.to_numeric(df["Population"])

    return df
