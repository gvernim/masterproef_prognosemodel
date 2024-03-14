import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Imputation imports
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
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

    sns.set_theme()
    sns.lineplot(df)
    sns.scatterplot(df_dates.iloc[:,0].where(df_dates['broken_record']), color="red")

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

    #Extend period with the values to the left with the same value as the broken value (Assumption: Broken sensors emit same data from point it is broken)
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
    df_periods.reset_index(drop=True)
    df_dates = df_reverse_dates[::-1]
    del df_dates['rolling_mean']

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
    #Mean/Median/Mode Imputation (Too simple for a complex dataset)

    #Rolling statistics Imputation (Not good for a period of broken data)

    #Linear Interpolation (Good for data with a linear trend, not for data with seasonal patterns)

    #Spline/polynomial Interpolation (Can create unrealistic fits for longer periods)

    #Linear Regression Imputation (Requires a secondary variable) //Unadapted copy
def replace_broken_records_knn(df, col_number, corr_col_number):
    # Drop missing values to fit the regression model
    df_imputed = df.copy()
    df_non_missing = df.dropna()

    # Instantiate the model
    model = LinearRegression()

    # Reshape data for model fitting (sklearn requires 2D array for predictors)
    X = df_non_missing['ad_spent'].values.reshape(-1, 1)
    Y = df_non_missing['sales'].values

    # Fit the model
    model.fit(X, Y)

    # Get indices of missing sales
    missing_sales_indices = df_imputed[df_imputed['sales'].isnull()].index

    # Predict missing sales values
    predicted_sales = model.predict(df_imputed.loc[missing_sales_indices, 'ad_spent'].values.reshape(-1, 1))

    # Fill missing sales with predicted values
    df_imputed.loc[missing_sales_indices, 'sales'] = predicted_sales

    # Plot the main line with markers
    df_imputed[['sales']].plot(style='.-', figsize=(12,8), title='Sales with Regression Imputation')

    # Add points where data was imputed with red color
    plt.scatter(missing_sales_indices, predicted_sales, color='red', label='Regression Imputation')

    # Set labels
    plt.xlabel('Time')
    plt.ylabel('Sales')

    plt.show()

    return df

    #K-Nearest Neighbours (Requires a secondary variable) //Unadapted copy
def replace_broken_records_knn(df, col_number, corr_col_number):
    # Initialize the KNN imputer with k=5
    imputer = KNNImputer(n_neighbors=3)

    # Apply the KNN imputer
    # Note: the KNNImputer requires 2D array-like input, hence the double brackets.
    df_imputed = df.copy()
    df_imputed[['sales', 'ad_spent']] = imputer.fit_transform(df_imputed[['sales', 'ad_spent']])

    # Create a matplotlib plot
    plt.figure(figsize=(12,8))
    df_imputed['sales'].plot(style='.-', label='Sales')

    # Add points where data was imputed
    imputed_indices = df[df['sales'].isnull()].index
    plt.scatter(imputed_indices, df_imputed.loc[imputed_indices, 'sales'], color='red', label='KNN Imputation')

    # Set title and labels
    plt.title('Sales with KNN Imputation')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    return df

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


    #STL Decomposition, does not work as required
def replace_broken_records_stl(df, col_number):

    # Fill missing values in the time series
    imputed_indices = df[df[df.columns[col_number]].isna()].index

    # Apply STL decompostion
    stl = STL(df[df.columns[col_number]].interpolate(), period=96, seasonal=729)
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

    #MSTL Decomposition
def replace_broken_records_mstl(df, col_number):

    # Fill missing values in the time series
    imputed_indices = df[df[df.columns[col_number]].isna()].index

    # Apply STL decompostion
    mstl = MSTL(df[df.columns[col_number]].interpolate(), periods=[96, 672], windows=[201, 201])
    res = mstl.fit()
    res.plot()
    plt.show()

    # Extract the seasonal and trend components
    seasonal_component_96 = res.seasonal["seasonal_96"]
    seasonal_component_672 = res.seasonal["seasonal_672"]

    # Create the deseasonalised series
    df_deseasonalised = df[df.columns[col_number]] - seasonal_component_96 - seasonal_component_672

    # Interpolate missing values in the deseasonalised series
    index_time = df_deseasonalised.index
    print(index_time)
    df_deseasonalised = df_deseasonalised.reset_index()
    print(df_deseasonalised)
    df_deseasonalised_imputed = df_deseasonalised.interpolate(method='cubic')

    df_deseasonalised_imputed.index = index_time
    print(df_deseasonalised_imputed)

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