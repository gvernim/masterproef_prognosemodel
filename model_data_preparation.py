import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

#Split data based on a manual date
def split_data(df, split_date):
    diff = datetime.timedelta(hours=1)

    train = df.loc[:split_date]

    test = df.loc[split_date+diff:]

    plt.plot(train, color = "black")
    plt.plot(test, color = "red")
    plt.title("Train/Test split Data")
    plt.ylabel(df.columns[0])
    plt.xlabel('TimeStamp')
    sns.set_theme()
    plt.show()
    return train, test

#Split based on a test percentage
def split_data_2(df, split_perc):
    total = len(df[df.columns[0]])
    split = int((1-split_perc)*total)
    df_col = df[df.columns[0]]
    train = df_col.iloc[:split]
    test = df_col.iloc[split:]

    plt.plot(train, color = "black")
    plt.plot(test, color = "red")
    plt.title("Train/Test split Data")
    plt.ylabel(df.columns[0])
    plt.xlabel('TimeStamp')
    sns.set_theme()
    plt.show()
    return train, test