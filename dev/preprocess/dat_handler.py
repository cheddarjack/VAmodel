import pandas as pd

def load_data(tick_data_path):
    tick_data = pd.read_parquet(tick_data_path)
    return tick_data

def save_data(tick_data, tick_data_path,):
    tick_data.to_parquet(tick_data_path)
 