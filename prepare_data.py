import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Determine lost and broken data
def show_missing_data(df):
    print('Data contains NaN values: ', df.isna().values.any())

    if df.isna().values.any():
        print('% missing data: \n', 100*df.isnull().sum()/len(df))

#Show possible faulty data per column
def find_missing_data(df, col_number):
    df_dates = pd.DataFrame(df[df.columns[col_number]], index=df.index)
    #df_dates['shifted_column'] = df_dates[df.columns[col_number]].shift(1)
    df_dates['rolling_mean'] = df[df.columns[col_number]].rolling(68).mean()

    df_dates['same_as_mean'] = df_dates[df.columns[col_number]] == df_dates['rolling_mean']

    #sns.lineplot(df_dates[df.columns[col_number]])
    #sns.lineplot(df_dates['same_as_mean'], color="red")

    #plt.show()

    df_reverse_dates = df_dates[::-1]

    df_periods = pd.DataFrame({'start_period': [], 'end_period': []})
    df_entry = pd.DataFrame({'start_period': [], 'end_period': []})
    in_period = False
    for index, row in df_reverse_dates.iterrows():
        if row['same_as_mean'] == True and in_period == False:
            in_period = True
            df_entry.loc[0, 'end_period'] = index
            faulty_value = row[df.columns[col_number]]
        elif row['same_as_mean'] == False and in_period == True and faulty_value == row[df.columns[col_number]]:
            df_reverse_dates.loc[index,'same_as_mean'] = True
        elif row['same_as_mean'] == False and in_period == True and faulty_value != row[df.columns[col_number]]:
            in_period = False
            df_entry.loc[0, 'start_period'] = index
            df_periods = df_periods._append(df_entry, ignore_index=True)

    df_periods = df_periods[::-1]
    df_periods.reset_index(drop=True)
    df_dates = df_reverse_dates[::-1]
    del df_dates['rolling_mean']

    print(df_dates)
    print('The following periods contain possible faulty data: \n', df_periods)

    sns.lineplot(df_dates[df.columns[col_number]])
    sns.lineplot(df_dates['same_as_mean'], color="red")

    plt.show()

    return df_dates, df_periods

#Make faulty records a NaN
def convert_broken_records_to_nan(df, col_number, df_dates, df_periods):
    df_dates.loc[df_dates['same_as_mean'] == True, df.columns[col_number]] = np.nan
    print(df_dates)
    df[df.columns[col_number]] = df_dates[df.columns[col_number]]
    print(df[df.columns[col_number]].tail())

    return df

#Replace the broken records
def replace_broken_records(df):
    return df