import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

def split_data_2(df, col_number, split_perc):
    return train_test_split(df[df.columns[col_number]], test_size=split_perc, random_state=0)