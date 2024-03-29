import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

def convert_datetime_index_to_prophet_df(df):
    for i in range(len(df.columns)-1):
        del df[df.columns[i+1]]

    df = df.reset_index().rename(columns={'TimeStamp':'ds', df.columns[0]:'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)

    return df