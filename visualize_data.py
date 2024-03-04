import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Alle columns tonen, slechte onduidelijke representatie
def visualize_columns(df):
    df.plot(x='TimeStamp',subplots=True)
    plt.tight_layout()
    plt.show()

#Eén column tonen, duidelijker te zien waar er mogelijk fouten zitten
def visualize_column(df, col_number):
    sns.lineplot(data=df, x='TimeStamp', y = df.columns[col_number])
    plt.ylabel(df.columns[col_number])
    plt.show()

#Column tonen in specifieke periode met start en einde
def visualize_column_period_start_end(df, col_number, start_period, end_period):
    sns.lineplot(data=df[start_period:end_period], x='TimeStamp', y = df.columns[col_number])
    plt.ylabel(df.columns[col_number])
    plt.show()

#Column tonen in specifieke periode met start en lengte periode
def visualize_column_period_start_length(df, col_number, start_period, length_period):
    end_period = start_period + length_period
    sns.lineplot(data=df[start_period:end_period], x='TimeStamp', y = df.columns[col_number])
    plt.ylabel(df.columns[col_number])
    plt.show()