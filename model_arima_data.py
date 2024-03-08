import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

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

def execute_arima(train, test, order):
    model = ARIMA(train, order=(1,1,4))
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.predict(start=0, end=len(test))
    print(predictions)
    predictions = pd.DataFrame(predictions, index=test.index)

    return model_fit, predictions


def execute_rolling_arima(train, test, order):
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    return model_fit, predictions