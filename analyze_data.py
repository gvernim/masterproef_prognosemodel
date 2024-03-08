import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose

#Correlation function
def correlation_between_columns(df, col_number_1, col_number_2):
    print('Correlation between ', df.columns[col_number_1], ' and ', df.columns[col_number_2],': ' , df[df.columns[col_number_1]].corr(df[df.columns[col_number_2]]))


#Decomposition analysis
def analyze_decomp_column(df, col_number):
    decompose = seasonal_decompose(df[df.columns[col_number]] ,model='additive', period=96)
    decompose.plot()
    plt.show()

def analyze_decomp_column_period_start_end(df, col_number, start_period, end_period):
    df = df[start_period:end_period]
    decompose = seasonal_decompose(df[df.columns[col_number]] ,model='additive', period=96)
    decompose.plot()
    plt.show()

def analyze_decomp_column_period_start_length(df, col_number, start_period, length_period):
    end_period = start_period + length_period
    df = df[start_period:end_period]
    decompose = seasonal_decompose(df[df.columns[col_number]] ,model='additive', period=96)
    decompose.plot()
    plt.show()

#Stationary analysis functions
#ACF for determing d-value
def show_plot_acf(df, col_number):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(df[df.columns[col_number]], ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

def show_plot_acf_1_diff(df, col_number):
    f= plt.figure()
    ax1 = f.add_subplot(121)
    ax1.plot(df[df.columns[col_number]].diff())

    ax2 = f.add_subplot(122)
    plot_acf(df[df.columns[col_number]].diff().dropna(), ax = ax2)
    plt.ylabel('Autocorrelation 1 Diff')
    plt.title('Autocorrelation Function (ACF)')

    plt.show()

def show_plot_acf_2_diff(df, col_number):
    f= plt.figure()
    ax1 = f.add_subplot(121)
    ax1.plot(df[df.columns[col_number]].diff().diff())

    ax2 = f.add_subplot(122)
    plot_acf(df[df.columns[col_number]].diff().diff().dropna(), ax = ax2)
    plt.ylabel('Autocorrelation 2 Diff')
    plt.title('Autocorrelation Function (ACF)')

    plt.show()

#Making sure of d-parameter
def adfuller_test(df, col_number):
    result = adfuller(df[df.columns[col_number]], autolag="AIC")
    print(f"Test Statistic: {result[0]}")
    print(f"P-value: {result[1]}")

    result = adfuller(df[df.columns[col_number]].diff().dropna(), autolag="AIC")
    print(f"Test Statistic 1 diff: {result[0]}")
    print(f"P-value: {result[1]}")

    result = adfuller(df[df.columns[col_number]].diff().diff().dropna(), autolag="AIC")
    print(f"Test Statistic 2 diff: {result[0]}")
    print(f"P-value: {result[1]}")

#Another metric for the d-parameter
def kpss_test(df, col_number):
    result = kpss(df[df.columns[col_number]])
    print(f"Test Statistic: {result[0]}")
    print(f"P-value: {result[1]}")

    result = kpss(df[df.columns[col_number]].diff().dropna())
    print(f"Test Statistic 1 diff: {result[0]}")
    print(f"P-value: {result[1]}")

    result = kpss(df[df.columns[col_number]].diff().diff().dropna())
    print(f"Test Statistic 2 diff: {result[0]}")
    print(f"P-value: {result[1]}")

def show_plot_pacf(df, col_number):
    # Plot PACF
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(df[df.columns[col_number]], ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

def show_plot_pacf_1_diff(df, col_number):
    f= plt.figure()
    ax1 = f.add_subplot(121)
    ax1.plot(df[df.columns[col_number]].diff())

    ax2 = f.add_subplot(122)
    plot_pacf(df[df.columns[col_number]].diff().dropna(), ax = ax2)
    plt.ylabel('Partial Autocorrelation 1 Diff')
    plt.title('Partial Autocorrelation Function (PACF)')

    plt.show()

def show_plot_pacf_2_diff(df, col_number):
    f= plt.figure()
    ax1 = f.add_subplot(121)
    ax1.plot(df[df.columns[col_number]].diff().diff())

    ax2 = f.add_subplot(122)
    plot_pacf(df[df.columns[col_number]].diff().diff().dropna(), ax = ax2)
    plt.ylabel('Partial Autocorrelation 2 Diff')
    plt.title('Partial Autocorrelation Function (PACF)')

    plt.show()

#Extra
def analyze_stat_column(df, col_number):
    rolling_mean = df[df.columns[col_number]].rolling(96).mean()
    rolling_std = df[df.columns[col_number]].rolling(96).std()
    adft = adfuller(df[df.columns[col_number]],autolag="AIC")
    stat_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print(stat_df)
    sns.set_theme()
    sns.lineplot(data=df, x='TimeStamp', y = df.columns[col_number])
    sns.lineplot(rolling_mean, color="red", label="Rolling Mean")
    sns.lineplot(rolling_std, color="black", label = "Rolling Standard Deviation")
    plt.title(df.columns[col_number] + ", Rolling Mean, Rolling Standard Deviation")
    plt.legend(loc="best")
    plt.show()

def analyze_stat_column_period_start_end(df, col_number, start_period, end_period):
    df = df[start_period:end_period]
    rolling_mean = df[df.columns[col_number]].rolling(7).mean()
    rolling_std = df[df.columns[col_number]].rolling(7).std()
    adft = adfuller(df[df.columns[col_number]],autolag="AIC")
    stat_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print(stat_df)
    sns.set_theme()
    sns.lineplot(data=df, x='TimeStamp', y = df.columns[col_number])
    sns.lineplot(rolling_mean, color="red", label="Rolling Mean")
    sns.lineplot(rolling_std, color="black", label = "Rolling Standard Deviation")
    plt.title(df.columns[col_number] + ", Rolling Mean, Rolling Standard Deviation")
    plt.legend(loc="best")
    plt.show()

def analyze_stat_column_period_start_length(df, col_number, start_period, length_period):
    end_period = start_period + length_period
    df = df[start_period:end_period]
    rolling_mean = df[df.columns[col_number]].rolling(7).mean()
    rolling_std = df[df.columns[col_number]].rolling(7).std()
    adft = adfuller(df[df.columns[col_number]],autolag="AIC")
    stat_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print(stat_df)
    sns.set_theme()
    sns.lineplot(data=df, x='TimeStamp', y = df.columns[col_number])
    sns.lineplot(rolling_mean, color="red", label="Rolling Mean")
    sns.lineplot(rolling_std, color="black", label = "Rolling Standard Deviation")
    plt.title(df.columns[col_number] + ", Rolling Mean, Rolling Standard Deviation")
    plt.legend(loc="best")
    plt.show()

#Autocorrelation analysis
def analyze_ac_column(df, col_number):
    lag = 1
    autocorrelation_lag1 = df[df.columns[col_number]].autocorr(lag=lag)
    lag *= 4
    autocorrelation_lag_hour = df[df.columns[col_number]].autocorr(lag=lag)
    lag *= 24
    autocorrelation_lag_day = df[df.columns[col_number]].autocorr(lag=lag)

    lag_week = lag*7
    lag_month = lag*30
    lag_year = lag*365
    autocorrelation_lag_week = df[df.columns[col_number]].autocorr(lag=lag_week)
    autocorrelation_lag_month = df[df.columns[col_number]].autocorr(lag=lag_month)
    autocorrelation_lag_year = df[df.columns[col_number]].autocorr(lag=lag_year)

    print("One 15 minute Lag: ", autocorrelation_lag1)
    print("One hour Lag: ", autocorrelation_lag_hour)
    print("One day Lag: ", autocorrelation_lag_day)
    print("One week Lag: ", autocorrelation_lag_week)
    print("One month Lag: ", autocorrelation_lag_month)
    print("One year Lag: ", autocorrelation_lag_year)

def analyze_ac_column_period_start_end(df, col_number, start_period, end_period):
    #end_period = start_period + length_period
    df = df[start_period:end_period]
    print(df.shape[0])
    lag = 1
    autocorrelation_lag1 = df[df.columns[col_number]].autocorr(lag=lag)
    print("One 15 minute Lag: ", autocorrelation_lag1)
    lag *= 4
    autocorrelation_lag_hour = df[df.columns[col_number]].autocorr(lag=lag)
    print("One hour Lag: ", autocorrelation_lag_hour)
    lag *= 24
    autocorrelation_lag_day = df[df.columns[col_number]].autocorr(lag=lag)
    print("One day Lag: ", autocorrelation_lag_day)

    lag_week = lag*7
    if df.shape[0]/10 > lag_week:
        autocorrelation_lag_week = df[df.columns[col_number]].autocorr(lag=lag_week)
        print("One week Lag: ", autocorrelation_lag_week)
        lag_month = lag*30
        if df.shape[0]/10 > lag_month:
            autocorrelation_lag_month = df[df.columns[col_number]].autocorr(lag=lag_month)
            print("One month Lag: ", autocorrelation_lag_month)
            lag_year = lag*365
            if df.shape[0]/10 > lag_year:
                autocorrelation_lag_year = df[df.columns[col_number]].autocorr(lag=lag_year)
                print("One year Lag: ", autocorrelation_lag_year)

def analyze_ac_column_period_start_length(df, col_number, start_period, length_period):
    end_period = start_period + length_period
    df = df[start_period:end_period]
    lag = 1
    autocorrelation_lag1 = df[df.columns[col_number]].autocorr(lag=lag)
    print("One 15 minute Lag: ", autocorrelation_lag1)
    lag *= 4
    autocorrelation_lag_hour = df[df.columns[col_number]].autocorr(lag=lag)
    print("One hour Lag: ", autocorrelation_lag_hour)
    lag *= 24
    autocorrelation_lag_day = df[df.columns[col_number]].autocorr(lag=lag)
    print("One day Lag: ", autocorrelation_lag_day)

    lag_week = lag*7
    if df.shape[0] > lag_week:
        autocorrelation_lag_week = df[df.columns[col_number]].autocorr(lag=lag_week)
        print("One week Lag: ", autocorrelation_lag_week)
        lag_month = lag*30
        if df.shape[0] > lag_month:
            autocorrelation_lag_month = df[df.columns[col_number]].autocorr(lag=lag_month)
            print("One month Lag: ", autocorrelation_lag_month)
            lag_year = lag*365
            if df.shape[0] > lag_year:
                autocorrelation_lag_year = df[df.columns[col_number]].autocorr(lag=lag_year)
                print("One year Lag: ", autocorrelation_lag_year)