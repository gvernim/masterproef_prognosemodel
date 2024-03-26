import pandas as pd
import numpy as np
import datetime
import holidays

import matplotlib.pyplot as plt
import seaborn as sns

import timeit

import csv
import json

#Default variables
filename = 'Merelbeke Energie.csv'

#Load data from file into dataframe
def load_csv_data_in_df(filename = 'Merelbeke Energie.csv'):
    df = pd.read_csv(filename, sep=';')
    df = change_header(df)
    df = format_data(df, 'csv')
    return df

def load_json_data_in_df(filename = 'Merelbeke Energie_2.json'):
    f = open(filename)
    data = json.load(f)

    df = pd.DataFrame(data['DetailData'])

    #df = change_header(df)
    df = format_data(df, 'json')
    return df

def load_extra_data_in_df(filename = 'Merelbeke Weerdata.csv'):
    df = pd.read_csv(filename, sep=',')

    df['TimeStamp'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S')
    df.index = df['TimeStamp']
    df.index = df.index.tz_localize('Europe/Brussels', ambiguous='infer')
    df = df.drop_duplicates()

    del df['TimeStamp']
    del df['datetime']

    return df

def change_header(df):
    for col_name in df.columns:
        df = df.rename(columns={col_name: col_name.replace('ATS Groep / Merelbeke / ', '')})
    return df

def format_data(df, type):
    #Convert string into datetime
    if type=='csv':
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%dT%H:%M:%S.000%z', utc=True)
    elif type == 'json':
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%dT%H:%M:%S%z', utc=True)

    #Make timestamp the index and convert to correct timezone
    df.index = df['TimeStamp']
    df.index = df.index.tz_convert('Europe/Brussels')
    df = df.drop_duplicates()

    #Delete unused columns
    del df['TimeStamp']
    return df

def change_quarterly_index_to_hourly(df):
    df_new = df.groupby(pd.Grouper(level='TimeStamp', freq='h')).sum()
    return df_new

def give_bank_holidays(start_date, end_date, manual=True):
    date_range = pd.date_range(start_date, end_date, freq='D')

    if not manual:
        bel_holidays = holidays.country_holidays('BE')
        national_holidays = [date for date in date_range if (date in bel_holidays)]
    else:
        bel_holidays = [pd.to_datetime('2022-01-01 00:00:00+01:00'), pd.to_datetime('2022-01-03 00:00:00+01:00'), pd.to_datetime('2022-04-18 00:00:00+01:00'), pd.to_datetime('2022-05-01 00:00:00+01:00'), pd.to_datetime('2022-05-02 00:00:00+01:00'), pd.to_datetime('2022-05-26 00:00:00+01:00'), pd.to_datetime('2022-05-27 00:00:00+01:00'), pd.to_datetime('2022-06-06 00:00:00+01:00'), pd.to_datetime('2022-07-21 00:00:00+01:00'), pd.to_datetime('2022-08-15 00:00:00+01:00'), pd.to_datetime('2022-11-01 00:00:00+01:00'), pd.to_datetime('2022-11-11 00:00:00+01:00'), pd.to_datetime('2022-12-25 00:00:00+01:00'), pd.to_datetime('2022-12-26 00:00:00+01:00'), pd.to_datetime('2023-01-02 00:00:00+01:00'), pd.to_datetime('2023-04-10 00:00:00+01:00'), pd.to_datetime('2023-05-01 00:00:00+01:00'), pd.to_datetime('2023-05-18 00:00:00+01:00'), pd.to_datetime('2023-05-19 00:00:00+01:00'), pd.to_datetime('2023-05-29 00:00:00+01:00'), pd.to_datetime('2023-07-21 00:00:00+01:00'), pd.to_datetime('2023-08-15 00:00:00+01:00'), pd.to_datetime('2023-11-01 00:00:00+01:00'), pd.to_datetime('2023-11-11 00:00:00+01:00'), pd.to_datetime('2023-12-25 00:00:00+01:00'), pd.to_datetime('2023-12-26 00:00:00+01:00'), pd.to_datetime('2024-01-01 00:00:00+01:00'), pd.to_datetime('2024-04-01 00:00:00+01:00'), pd.to_datetime('2024-05-01 00:00:00+01:00'), pd.to_datetime('2024-05-09 00:00:00+01:00'), pd.to_datetime('2024-05-20 00:00:00+01:00'), pd.to_datetime('2024-07-21 00:00:00+01:00'), pd.to_datetime('2024-08-15 00:00:00+01:00'), pd.to_datetime('2024-08-16 00:00:00+01:00'), pd.to_datetime('2024-11-01 00:00:00+01:00'), pd.to_datetime('2024-11-11 00:00:00+01:00'), pd.to_datetime('2024-12-25 00:00:00+01:00')]

    national_holidays = [date for date in date_range if (date in bel_holidays)]

    return pd.DataFrame(national_holidays, columns=['Date'])

def give_bank_holidays_quarterly(start_date, end_date):
    df_holidays = give_bank_holidays(start_date, end_date)
    for index, row in df_holidays.iterrows():
        dates = pd.DataFrame({'Date': pd.date_range(row['Date'], periods=96, freq='15min')})
        df_holidays = pd.concat([df_holidays, dates]).drop_duplicates().sort_values(by='Date').reset_index(drop=True)
    return df_holidays

def give_bank_holidays_hourly(start_date, end_date):
    df_holidays = give_bank_holidays(start_date, end_date)
    for index, row in df_holidays.iterrows():
        dates = pd.DataFrame({'Date': pd.date_range(row['Date'], periods=24, freq='h')})
        df_holidays = pd.concat([df_holidays, dates]).drop_duplicates().sort_values(by='Date').reset_index(drop=True)
    return df_holidays

#Test functie, unused
def timer_func(func):
    start = timeit.default_timer()
    value = func()
    runtime = timeit.default_timer() - start
    msg = "{func} took {time} seconds to complete its execution."
    print(msg.format(func = func.__name__,time = runtime))

#timer_func(load_data)