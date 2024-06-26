import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    return df

def create_lag_features(df, lags):
    df = df.copy()
    for i in lags:
        df['lag' + str(i)] = df[df.columns[0]].shift(i)
        df['lag' + str(i)] = df['lag' + str(i)].fillna(0)
    return df