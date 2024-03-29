import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

def split_data_hour_2_weeks(df, start_date, split_date):

    diff = datetime.timedelta(hours=1)
    validation_period = datetime.timedelta(days=13, hours=23)

    train = df[start_date:split_date-diff]
    validation = df[split_date:split_date+validation_period]

    #plt.plot(train[validation.columns[0]], color = "black")
    #plt.plot(validation[validation.columns[0]], color = "red")
    #plt.title("Train/Validation split Data")
    #plt.ylabel(df.columns[0])
    #plt.xlabel('TimeStamp')
    #sns.set_theme()
    #plt.show()
    return train, validation

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

def execute_arima(train, validation, order, seasonal_order):
    print(validation)
    model = ARIMA(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.predict(start=validation.index[0], end=validation.index[-1])
    print(predictions)
    #predictions = pd.DataFrame(predictions, index=validation.index)

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