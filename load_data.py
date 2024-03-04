import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import timeit
import time

import csv

#Default variables
filename = 'Merelbeke Energie.csv'

#Load data from file into dataframe
def load_data_in_df(filename = 'Merelbeke Energie.csv'):
    df = pd.read_csv(filename, sep=';')
    df = change_header(df)
    df = format_data(df)
    return df

def change_header(df):
    for col_name in df.columns:
        df = df.rename(columns={col_name: col_name.replace('ATS Groep / Merelbeke / ', '')})
    return df

def format_data(df):
    #Convert string into datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%dT%H:%M:%S.000%z', utc=True)
    #df['TimeStamp'] = datetime.strptime(df['TimeStamp'], format) #Only for single entry strings

    #Make timestamp the index
    df.index = df['TimeStamp']

    #Delete unused columns
    del df['TimeStamp']
    del df['UnitOfMeasurement']
    return df

#Test functie, unused
def timer_func(func):
    start = timeit.default_timer()
    value = func()
    runtime = timeit.default_timer() - start
    msg = "{func} took {time} seconds to complete its execution."
    print(msg.format(func = func.__name__,time = runtime))

#load_data_in_df()
#timer_func(load_data)