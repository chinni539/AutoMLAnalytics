import pandas as pd

def is_numeric(df, column):

    return pd.api.types.is_numeric_dtype(df[column])


def is_categorical(df, column):

    return pd.api.types.is_string_dtype(df[column])