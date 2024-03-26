import pandas as pd
import numpy as np
import math
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

#Imputation imports
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor

from statsmodels.imputation.mice import MICE, MICEData
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import seasonal_decompose

#Determine if there is lost and broken data and the percentages
def show_missing_data(df):
    print('Data contains NaN values: ', df.isna().values.any())

    if df.isna().values.any():
        print('% missing data: \n', 100*df.isnull().sum()/len(df))

def show_missing_data_column(df, col_number):
    print(df.columns[col_number], ' contains NaN values: ', df[df.columns[col_number]].isna().values.any())

    if df[df.columns[col_number]].isna().values.any():
        print('% missing data: \n', 100*df[df.columns[col_number]].isnull().sum()/len(df[df.columns[col_number]]))


#Show possible extreme value points
def find_missing_data_points(df):
    df_dates = pd.DataFrame(df, index=df.index)

    col_pos_mean = df_dates[df_dates>=0].mean()
    col_neg_mean = df_dates[df_dates<0].mean()

    df_dates['broken_record'] = df_dates.iloc[:, 0].between(50*col_neg_mean.iloc[0], 100*col_pos_mean.iloc[0])
    df_dates['broken_record'] = ~df_dates['broken_record']

    #sns.set_theme()
    #sns.lineplot(df)
    #sns.scatterplot(df_dates.iloc[:,0].where(df_dates['broken_record']), color="red")

    plt.show()

    return df_dates

def find_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   outliers = df[((df<(q1-4*IQR)) | (df>(q3+4*IQR)))]

   sns.set_theme()
   sns.lineplot(df)
   sns.lineplot(outliers, color="red")
   plt.show()

   return outliers

#Show possible missing periods of data per column
def find_missing_data_periods(df, rolling_records=68):
    df_dates = pd.DataFrame(df, index=df.index)

    #Shifts one to the left and compares but doesnt work properly
    #df_dates['shifted_column'] = df_dates[df.columns[col_number]].shift(1)

    #Rolling mean takes 
    df_dates['rolling_mean'] = df.rolling(rolling_records).mean()

    df_dates['broken_record'] = df_dates.iloc[:, 0] == df_dates['rolling_mean']

    #Visualize found periods with rolling mean (Incomplete cause it considers the values to its left so the initial part of the broken period isnt included)
    #sns.set_theme()
    #sns.lineplot(df_dates.iloc[:, 0])
    #sns.lineplot(df_dates['broken_record'], color="red")
    #plt.show()

    #Extend period with the values to the left with the same value as the broken value (Assumption: Broken sensors/software record same data from point it is broken)
    df_reverse_dates = df_dates[::-1]

    df_periods = pd.DataFrame({'start_period': [], 'end_period': []}, dtype=object)
    df_entry = pd.DataFrame({'start_period': [], 'end_period': []}, dtype=object)
    in_period = False
    for index, row in df_reverse_dates.iterrows():
        if row['broken_record'] == True and in_period == False:
            in_period = True
            df_entry.loc[0, 'end_period'] = index
            faulty_value = row.iloc[0]
        elif row['broken_record'] == False and in_period == True and faulty_value == row.iloc[0]:
            df_reverse_dates.loc[index,'broken_record'] = True
        elif row['broken_record'] == False and in_period == True and faulty_value != row.iloc[0]:
            in_period = False
            df_entry.loc[0, 'start_period'] = index
            df_periods = df_periods._append(df_entry, ignore_index=True)

    df_periods = df_periods[::-1]
    df_periods = df_periods.reset_index(drop=True)
    df_dates = df_reverse_dates[::-1]
    del df_dates['rolling_mean']

    for index, row in df_periods.iterrows():
        if len(df_dates[row.iloc[0]:row.iloc[1]]) <= 75 and df_dates.loc[row.iloc[0], df_dates.columns[0]] <= 0.4:
            df_dates.loc[row.iloc[0]:row.iloc[1], 'broken_record'] = False
            df_periods.drop(index, inplace=True)



    #Show periods with broken data

    #sns.set_theme()
    #sns.lineplot(df_dates.iloc[:,0])
    #sns.lineplot(df_dates['broken_record'], color="red")

    #plt.show()

    return df_dates, df_periods


#Make faulty records a NaN
def convert_broken_records_to_nan(df, col_number, df_dates, df_periods):
    df_dates.loc[df_dates['broken_record'] == True, df.columns[col_number]] = np.nan
    df[df.columns[col_number]] = df_dates[df.columns[col_number]]

    return df[df.columns[col_number]]

def find_largest_period_without_nan(df):
    df_test = df
    a = df_test.values
    m = np.concatenate(( [True], np.isnan(a), [True] ))
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)
    start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]

    return df_test.iloc[start:stop-1], start, stop


#Replace the broken records
def replace_broken_records_custom(df):
    result_day = df.groupby(df.index.hour).mean()
    result_week = df.groupby(df.index.weekday).mean()
    result_month = df.groupby(df.index.month).mean()

    #mean_hours = result_day.mean()
    mean_days = result_week.mean()
    mean_months = result_month.mean()

    imputed_indices = df[df.isna()].index

    predicted_values = pd.DataFrame({'hour': imputed_indices.hour, 'weekday': imputed_indices.weekday, 'month': imputed_indices.month} , index=imputed_indices)

    for index, row in predicted_values.iterrows():
        prediction = result_day.iloc[row.iloc[0]]*(result_week.iloc[row.iloc[1]]/mean_days)*(result_month.iloc[row.iloc[2]-1]/mean_months)
        df.loc[index] = prediction
        predicted_values.loc[index, 'pred_value'] = prediction

    df.plot()
    plt.scatter(imputed_indices, predicted_values['pred_value'], color='red', label='Custom Imputation')

    plt.show()

    return df

    #Mean/Median/Mode Imputation (Too simple for a complex dataset)

    #Rolling statistics Imputation (Not good for a period of broken data)

    #Linear Interpolation (Good for data with a linear trend, not for data with seasonal patterns)

    #Spline/polynomial Interpolation (Can create unrealistic fits for longer periods)

    #Linear Regression Imputation (Requires features) //Works, but not correct
def replace_broken_records_linear_regression(df):
    # Drop missing values to fit the regression model
    df_imputed = df.copy()
    df_non_missing = df.dropna()

    # Instantiate the model
    model = LinearRegression()

    # Reshape data for model fitting (sklearn requires 2D array for predictors)
    X = df_non_missing[df.columns[1]].values.reshape(-1, 1)
    Y = df_non_missing[df.columns[0]].values

    # Fit the model
    model.fit(X, Y)

    # Get indices of missing
    missing_indices = df_imputed[df_imputed[df.columns[0]].isnull()].index

    # Predict missing values
    predicted = model.predict(df_imputed.loc[missing_indices, df.columns[1]].values.reshape(-1, 1))

    # Fill missing with predicted values
    df_imputed.loc[missing_indices, df.columns[0]] = predicted

    # Plot the main line with markers
    df_imputed[[df.columns[0]]].plot(style='.-', figsize=(12,8), title='Data with Regression Imputation')

    # Add points where data was imputed with red color
    plt.scatter(missing_indices, predicted, color='red', label='Regression Imputation')

    # Set labels
    plt.xlabel('TimeStamp')
    plt.ylabel(df.columns[0])

    plt.show()

    return df

    #K-Nearest Neighbours (Requires features to impute data) //Works better than linear regression
def replace_broken_records_knn(df, col_number, corr_col_number):
    # Initialize the KNN imputer with k=5
    imputer = KNNImputer(n_neighbors=3)

    # Apply the KNN imputer
    # Note: the KNNImputer requires 2D array-like input, hence the double brackets.
    df_imputed = df.copy()
    df_imputed[[df.columns[col_number], df.columns[corr_col_number]]] = imputer.fit_transform(df_imputed[[df.columns[col_number], df.columns[corr_col_number]]])

    # Create a matplotlib plot
    plt.figure(figsize=(12,8))
    df_imputed[df.columns[col_number]].plot(style='.-', label=df.columns[col_number])

    # Add points where data was imputed
    imputed_indices = df[df[df.columns[col_number]].isna()].index
    plt.scatter(imputed_indices, df_imputed.loc[imputed_indices, df.columns[col_number]], color='red', label='KNN Imputation')

    # Set title and labels
    plt.title('KNN Imputation')
    plt.xlabel(df.columns[corr_col_number])
    plt.ylabel(df.columns[col_number])
    plt.legend()
    plt.show()
    return df_imputed

    #Seasonal decompose Decomposition, does not work as required
def replace_broken_records_seasonal_decompose(df, col_number):

    # Fill missing values in the time series
    imputed_indices = df[df[df.columns[col_number]].isna()].index

    # Apply STL decompostion
    stl = seasonal_decompose(df[df.columns[col_number]].interpolate(), period=96, model="multiplicative")
    res = stl.fit()
    res.plot()
    plt.show()

    # Extract the seasonal and trend components
    seasonal_component = res.seasonal

    # Create the deseasonalised series
    df_deseasonalised = df[df.columns[col_number]] - seasonal_component

    # Interpolate missing values in the deseasonalised series
    df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")

    # Add the seasonal component back to create the final imputed series
    df_imputed = df_deseasonalised_imputed + seasonal_component

    # Update the original dataframe with the imputed value
    df.loc[imputed_indices, df.columns[col_number]] = df_imputed[imputed_indices]

    # Plot the series using pandas
    plt.figure(figsize=[12, 6])
    df[df.columns[col_number]].plot(style='.-',  label=df.columns[col_number])

    plt.scatter(imputed_indices, df[df.columns[col_number]].loc[imputed_indices], color='red')

    plt.title("STL Imputation")
    plt.ylabel(df.columns[col_number])
    plt.xlabel("TimeStamp")
    plt.show()
    return df


    #STL Decomposition, does not work as required for long periods
def replace_broken_records_stl(df, col_number):

    seasonal_calc = math.floor(len(df)/96)
    if seasonal_calc % 2 == 0:
        seasonal_calc -= 1

    trend_calc = round(3*seasonal_calc)
    if trend_calc % 2 == 0:
        trend_calc -= 1

    print('Seasonal: ', seasonal_calc)
    print('Trend: ', trend_calc)

    # Fill missing values in the time series
    imputed_indices = df[df[df.columns[col_number]].isna()].index
    #df[df.columns[col_number]] = df[df.columns[col_number]].fillna(0)

    # Apply STL decompostion
    stl = STL(df[df.columns[col_number]].interpolate(), period=96, trend=trend_calc, seasonal=seasonal_calc)
    res = stl.fit()
    res.plot()
    plt.show()

    # Extract the seasonal and trend components
    seasonal_component = res.seasonal

    # Create the deseasonalised series
    df_deseasonalised = df[df.columns[col_number]] - seasonal_component

    # Interpolate missing values in the deseasonalised series
    df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")
    #df_deseasonalised_imputed = df_deseasonalised
    #df_deseasonalised_imputed.loc[imputed_indices] = 0

    # Add the seasonal component back to create the final imputed series
    df_imputed = df_deseasonalised_imputed + seasonal_component

    # Update the original dataframe with the imputed value
    df.loc[imputed_indices, df.columns[col_number]] = df_imputed[imputed_indices]

    # Plot the series using pandas
    plt.figure(figsize=[12, 6])
    df[df.columns[col_number]].plot(style='.-',  label=df.columns[col_number])

    plt.scatter(imputed_indices, df[df.columns[col_number]].loc[imputed_indices], color='red')

    plt.title("STL Imputation")
    plt.ylabel(df.columns[col_number])
    plt.xlabel("TimeStamp")
    plt.show()
    return df

    #MSTL Decomposition, doesnt work as required for long periods
def replace_broken_records_mstl(df, col_number):

    seasonal_calc_1 = math.floor(len(df)/96)
    seasonal_calc_2 = math.floor(len(df)/96)
    if seasonal_calc_1 % 2 == 0:
        seasonal_calc_1 -= 1
    if seasonal_calc_2 % 2 == 0:
        seasonal_calc_2 -= 1

    # Fill missing values in the time series
    imputed_indices = df[df[df.columns[col_number]].isna()].index
    df[df.columns[col_number]] = df[df.columns[col_number]].fillna(0)

    # Apply STL decompostion
    mstl = MSTL(df[df.columns[col_number]].interpolate(), periods=[96, 672], windows=[seasonal_calc_1, seasonal_calc_2])
    res = mstl.fit()
    res.plot()
    plt.show()

    # Extract the seasonal and trend components
    seasonal_component_96 = res.seasonal["seasonal_96"]
    seasonal_component_672 = res.seasonal["seasonal_672"]

    # Create the deseasonalised series
    df_deseasonalised = df[df.columns[col_number]] - seasonal_component_96 - seasonal_component_672

    # Interpolate missing values in the deseasonalised series
    df_deseasonalised_imputed = df_deseasonalised.interpolate(method='linear')

    # Add the seasonal component back to create the final imputed series
    df_imputed = df_deseasonalised_imputed + seasonal_component_96 + seasonal_component_672

    # Update the original dataframe with the imputed value
    df.loc[imputed_indices, df.columns[col_number]] = df_imputed[imputed_indices]

    # Plot the series using pandas
    plt.figure(figsize=[12, 6])
    df[df.columns[col_number]].plot(style='.-',  label=df.columns[col_number])

    plt.scatter(imputed_indices, df[df.columns[col_number]].loc[imputed_indices], color='red')

    plt.title("MSTL Imputation")
    plt.ylabel(df.columns[col_number])
    plt.xlabel("TimeStamp")
    plt.show()
    return df

def replace_broken_record_mice(df):
    return df

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
    diff = datetime.timedelta(hours=1)
    df_col = df[df.columns[0]]
    train = df_col.iloc[:split]
    test = df_col.iloc[split:]

    plt.plot(train[train.columns[0]], color = "black")
    plt.plot(test[test.columns[0]], color = "red")
    plt.title("Train/Test split Data")
    plt.ylabel(df.columns[0])
    plt.xlabel('TimeStamp')
    sns.set_theme()
    plt.show()
    return train, test