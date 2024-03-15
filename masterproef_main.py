#Import standard libraries
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

#Import statistical analysis
from sklearn.metrics import mean_squared_error

#Import arima model
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

#Import custom classes
import load_data
import visualize_data
import prepare_data
import analyze_data
import model_data_preparation
import model_arima_data

#User variables
filename = 'Merelbeke Energie.csv'
column_number = 0
start_date_data = pd.to_datetime('2022-01-01 00:00:00+01:00')
end_date_data = pd.to_datetime('2024-01-01 00:00:00+01:00')
    #Vorm tijdperiodes '2021-12-31 23:00:00+00:00' (Startdate of data)
start_period = pd.to_datetime('2022-01-01 00:00:00+01:00')
end_period = pd.to_datetime('2023-01-01 00:00:00+01:00')
length_period = datetime.timedelta(days=90)

split_date = pd.to_datetime('2023-10-01 16:00:00+01:00')

#Data headers with number in brackets
    #Timestamp[0] ; PV Productie[1] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[2] ; NA Aankomst / ActiveEnergyProduction(Productie)[3] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[4] ; SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)[5] ;
    #SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[6] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)[7] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[8] ; Verbruik laadpalen(Consumptie)[9] ; TotalProductie[10] ; TotalConsumptie[11] ; UnitOfMeasurement[12] ;

#Start main
#1. Load and format data
df = load_data.load_data_in_df(filename)
#df_holidays = load_data.give_bank_holidays(start_date_data, end_date_data, True)
#df_holidays = load_data.give_bank_holidays_quarterly(start_date_data, end_date_data)
#df_holidays_hourly = load_data.give_bank_holidays_hourly(start_date_data, end_date_data)

#Data headers with number in brackets after format
    #PV Productie[0] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[1] ; NA Aankomst / ActiveEnergyProduction(Productie)[2] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[3] ; SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)[4] ;
    #SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[5] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)[6] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[7] ; Verbruik laadpalen(Consumptie)[8] ; TotalProductie[9] ; TotalConsumptie[10] ;

#Tariff2 not required
del df['SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)']
del df['SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)']

#Data headers with number in brackets after format
    #PV Productie[0] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[1] ; NA Aankomst / ActiveEnergyProduction(Productie)[2] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[3] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[4] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[5] ; Verbruik laadpalen(Consumptie)[6] ; TotalProductie[7] ; TotalConsumptie[8] ;

#End Load and format data

#2. Visualization data

#visualize_data.visualize_columns(df)
#analyze_data.correlation_between_columns(df, 0, 7)
#analyze_data.correlation_between_columns(df, 1, 8)
#analyze_data.correlation_between_columns(df, 2, 7)
#analyze_data.correlation_between_columns(df, 3, 7)
#analyze_data.correlation_between_columns(df, 4, 8)
#analyze_data.correlation_between_columns(df, 5, 8)
#analyze_data.correlation_between_columns(df, 6, 8)

#visualize_data.visualize_column(df, column_number)
#visualize_data.visualize_column_period_start_end(df, column_number, start_period, end_period)
#visualize_data.visualize_column_period_start_length(df, column_number, start_period, length_period)

#for x in range(len(df.columns)):
#    visualize_data.visualize_column(df, x)

#End visualization data

#3. Data preparation

#Data does not contain NaN values by default for incomplete records
#prepare_data.show_missing_data(df)

#Select column and make hourly version
df_col = df[df.columns[column_number]]
df_col_hour = load_data.change_quarterly_index_to_hourly(df_col)

#prepare_data.find_outliers_IQR(df_col)

#Adapted per column, 
if column_number==0:
    rolling_records = 68
else:
    rolling_records = 68

#Find faulty data
df_dates, df_periods = prepare_data.find_missing_data_periods(df_col, rolling_records)
df_dates_points = prepare_data.find_missing_data_points(df_col)

print('The following periods contain possible faulty data: \n', df_periods)

#Add both broken periods and points together
df_dates['broken_record'] = df_dates['broken_record'] | df_dates_points['broken_record']

#sns.set_theme()
#sns.lineplot(df_dates[df.columns[column_number]])
#sns.lineplot(df_dates['broken_record'], color="red")
#sns.scatterplot(df_dates_points[df.columns[column_number]].where(df_dates['broken_record']), color="red")
#plt.show()

#After check replace with NaN values
df_col = prepare_data.convert_broken_records_to_nan(df, column_number, df_dates, df_periods)

#Show percentage of missing data
prepare_data.show_missing_data_column(df, column_number)

df_period, start, stop = prepare_data.find_largest_period_without_nan(df_col)
print(df_period)
print(start)
print(stop)

#prepare_data.find_outliers_IQR(df.iloc[start:stop], column_number)

#sns.set_theme()
#sns.lineplot(df_dates[df.columns[column_number]])
#plt.show()

#prepare_data.show_missing_data_column(df, column_number)

#Prepare data for all columns
#for x in range(len(df.columns)):
    #Find faulty data (Separate function, since it might be incorrect)
#    df_dates, df_periods = prepare_data.find_missing_data_periods(df, x)
#    print('The following periods contain possible faulty data: \n', df_periods)
#    df_dates_points = prepare_data.find_missing_data_points(df, column_number)
#    df_dates['broken_record'] = df_dates['broken_record'] | df_dates_points['broken_record']

    #After check replace with NaN values
#    df = prepare_data.convert_broken_records_to_nan(df, x, df_dates, df_periods)
#    sns.set_theme()
#    sns.lineplot(df_dates[df.columns[column_number]])
#    plt.show()

#prepare_data.show_missing_data(df)

#Impute missing data

prepare_data.replace_broken_records_custom(df_col)

# Both dont work as required. Can impute the seasonal component somewhat but is not able to impute the trend/residu component
# Leaves an unrealistic imputation
#df = prepare_data.replace_broken_records_seasonal_decompose(df, column_number)
#df = prepare_data.replace_broken_records_stl(df, column_number)
#df = prepare_data.replace_broken_records_mstl(df, column_number)

#For each seperate period
#for index, row in df_periods.iterrows():
#    print(len(df.loc[row.iloc[0]:row.iloc[1]]))

#    if index != 0 and len(df.loc[row.iloc[0]:row.iloc[1]]) < 5000 and index != len(df_periods)-1:
#        first_good_date = df_periods.iloc[index-1, 1]
#        print('First: ', first_good_date)
#        last_good_date = df_periods.iloc[index+1, 0]
#        print('Last: ',last_good_date)
#        df.loc[first_good_date:last_good_date, :] = prepare_data.replace_broken_records_stl(df.loc[first_good_date:last_good_date, :], column_number)

#End Data preparation

#4. Analysis

#Decomposition analysis
#analyze_data.analyze_decomp_column(df.iloc[start:stop-1], column_number)
#analyze_data.analyze_decomp_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_decomp_column_period_start_length(df, column_number, start_period, length_period)


#Stationary analysis
    #Determine d-parameter => Which lag is largest (If unclear check differencing ACF)
    #Determine q-parameter => Number of Lags outside of blue zone
#analyze_data.show_plot_acf(df.iloc[start:stop-1], column_number)
#analyze_data.show_plot_acf_1_diff(df.iloc[start:stop-1], column_number)
#analyze_data.show_plot_acf_2_diff(df.iloc[start:stop-1], column_number)

    #Double-check d-parameter (For ADF: Under 0.05)
#analyze_data.adfuller_test(df.iloc[start:stop-1], column_number)
#analyze_data.kpss_test(df.iloc[start:stop-1], column_number)

    #Determine p-parameter => Which lag is largest (If unclear check next differential PACF)
#analyze_data.show_plot_pacf(df.iloc[start:stop-1], column_number)
#analyze_data.show_plot_pacf_1_diff(df.iloc[start:stop-1], column_number)
#analyze_data.show_plot_pacf_2_diff(df.iloc[start:stop-1], column_number)

#Extra functions (unused)
#analyze_data.analyze_stat_column(df.iloc[start:stop-1], column_number)
#analyze_data.analyze_stat_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_stat_column_period_start_length(df, column_number, start_period, length_period)


#Autocorrelation analysis (unused)
#analyze_data.analyze_ac_column(df.iloc[start:stop-1], column_number)
#analyze_data.analyze_ac_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_ac_column_period_start_length(df, column_number, start_period, length_period)

#print(df.describe())

#End analysis


#5. Model data preparation

#train, test = prepare_data.split_data(df.iloc[start:stop-1], column_number, split_date)

#train, test = prepare_data.split_data_2(df.iloc[start:stop-1], column_number, 0.33)

#End Model data preparation

#6. ARIMA Model

#Function to determine parameters
#model = auto_arima(train, m=96, seasonal=True, stepwise=True)

#Standard ARIMA Model
order=(1,1,4)
#1,1,4 AIC=       1,1,5: AIC=     1,2,3: AIC=
#model_fit, predictions = model_arima_data.execute_rolling_arima(train, test, order)


#Rolling Forecast ARIMA model
#1,1,4 AIC=       1,1,5: AIC=     1,2,3: AIC=
#model_fit, predictions = model_arima_data.execute_rolling_arima(train, test, order)

#plt.plot(train, color = "black")
#plt.plot(test, color = "red")
#plt.plot(predictions, color = "blue")
#plt.title("Train/Test split Data")
#plt.ylabel(df.columns[column_number])
#plt.xlabel('TimeStamp')
#plt.show()

#End ARIMA Model

#7. Evaluation

#rmse = sqrt(mean_squared_error(test, predictions))
#print("RMSE: ", rmse)

#End Evaluation

#End main