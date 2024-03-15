import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Split data based on a manual date
def split_data(df, col_number, split_date):
    train = df.loc[:split_date, df.columns[col_number]]
    #train['train'] = train.loc[:,df.columns[col_number]]


    test = df.loc[split_date:, df.columns[col_number]]
    #test['test'] = test[df.columns[col_number]]
    #del test[df.columns[col_number]]


    plt.plot(train, color = "black")
    plt.plot(test, color = "red")
    plt.title("Train/Test split Data")
    plt.ylabel(df.columns[col_number])
    plt.xlabel('TimeStamp')
    sns.set_theme()
    plt.show()
    return train, test

#Split based on a test percentage
def split_data_2(df, col_number, split_perc):
    total = len(df[df.columns[col_number]])
    split = int((1-split_perc)*total)
    df_col = df[df.columns[col_number]]
    train = df_col.iloc[:split]
    test = df_col.iloc[split:]

    plt.plot(train, color = "black")
    plt.plot(test, color = "red")
    plt.title("Train/Test split Data")
    plt.ylabel(df.columns[col_number])
    plt.xlabel('TimeStamp')
    sns.set_theme()
    plt.show()
    return train, test