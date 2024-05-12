import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

def convert_datetime_index_to_prophet_df(df):

    df = df.reset_index().rename(columns={'TimeStamp':'ds', df.columns[0]:'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)

    return df

def create_lag_features(df, lags):
    df = df.copy()
    for i in lags:
        df['lag' + str(i)] = df[df.columns[0]].shift(i)
        df['lag' + str(i)] = df['lag' + str(i)].fillna(0)
    return df