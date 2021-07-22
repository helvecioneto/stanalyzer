import pandas as pd

def read_file(path=None):
    return pd.read_pickle(path)