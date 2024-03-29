#Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

#Import statistical analysis
from sklearn.metrics import mean_squared_error

#Import arima model
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.arima.model import ARIMA

#Import custom classes
import load_data
import visualize_data
import prepare_data
import analyze_data
import model_data_preparation
import model_arima_data

#User variables
filename_1 = 'Merelbeke Energie.csv'
filename_2 = 'Merelbeke Energie_2.json'
column_number = 0
start_date_data = pd.to_datetime('2022-01-01 00:00:00+01:00')
end_date_data = pd.to_datetime('2023-12-31 23:45:00+01:00')

start_date_data_json = pd.to_datetime('2022-01-01 00:00:00+01:00')
end_date_data_json = pd.to_datetime('2024-02-29 23:45:00+01:00')

    #Vorm tijdperiodes '2021-12-31 23:00:00+00:00' (Startdate of data)
start_period = pd.to_datetime('2023-01-01 00:00:00+01:00')
end_period = pd.to_datetime('2024-01-13 23:00:00+01:00')
length_period = datetime.timedelta(days=90)

start_test_set = pd.to_datetime('2024-01-01 00:00:00+01:00')
end_test_set_2_weeks = pd.to_datetime('2024-01-14 23:00:00+01:00')
end_test_set_month = pd.to_datetime('2024-01-31 23:00:00+01:00')
end_test_set_2_month = pd.to_datetime('2023-12-31 23:00:00+01:00')

split_date = pd.to_datetime('2023-12-18 00:00:00+01:00')

#Data headers with number in brackets
    #Timestamp[0] ; PV Productie[1] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[2] ; NA Aankomst / ActiveEnergyProduction(Productie)[3] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[4] ; SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)[5] ;
    #SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[6] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)[7] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[8] ; Verbruik laadpalen(Consumptie)[9] ; TotalProductie[10] ; TotalConsumptie[11] ; UnitOfMeasurement[12] ;



#Start main
#1. Load and format data
df = load_data.load_csv_data_in_df(filename_1)
del df['UnitOfMeasurement']

df_2024 = load_data.load_json_data_in_df(filename_2)
df_2024.columns = df.columns

df_weer = load_data.load_extra_data_in_df()

df_weer_test_set_2_weeks = df_weer.loc[start_test_set:end_test_set_2_weeks]
df_weer_test_set_month = df_weer.loc[start_test_set:end_test_set_month]
df_weer_test_set_2_month = df_weer.loc[start_test_set:end_test_set_2_month]

#Select column and make hourly version
df_col = pd.DataFrame(df_2024[df_2024.columns[column_number]], index=df_2024.index)
df_col_hour = load_data.change_quarterly_index_to_hourly(df_col)

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
del df_2024['SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)']
del df_2024['SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)']

#Data headers with number in brackets after format
    #PV Productie[0] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[1] ; NA Aankomst / ActiveEnergyProduction(Productie)[2] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[3] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[4] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[5] ; Verbruik laadpalen(Consumptie)[6] ; TotalProductie[7] ; TotalConsumptie[8] ;

#End Load and format data



#2. Visualization data

#visualize_data.visualize_columns(df)

#analyze_data.correlation_between_columns(df, df_weer, 0, 0)    #Very weak
#analyze_data.correlation_between_columns(df, df_weer, 0, 1)    #Weak negative
#analyze_data.correlation_between_columns(df, df_weer, 0, 2)    #Very weak
#analyze_data.correlation_between_columns(df, df_weer, 0, 3)    #Weak negative
analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 4)     #Very strong correlation

#visualize_data.visualize_column_period_start_end(df, column_number, start_period, end_period)
#visualize_data.visualize_column_period_start_length(df, column_number, start_period, length_period)

#End visualization data



#3. Data preparation

#Adapted per column, 
if column_number==0:
    rolling_records = 32
else:
    rolling_records = 68

#Find faulty data
df_dates, df_periods = prepare_data.find_missing_data_periods(df_col, rolling_records)
df_dates_points = prepare_data.find_missing_data_points(df_col)

print('The following periods contain possible faulty data: \n', df_periods)

#Add both broken periods and points together
df_dates['broken_record'] = df_dates['broken_record'] | df_dates_points['broken_record']

#Convert found faulty data to 
df_dates_hour = load_data.change_quarterly_index_to_hourly(df_dates)
df_dates_hour['broken_record'] = np.where(df_dates_hour['broken_record'] >= 1, True, False)

#After check replace with NaN values
df_col = prepare_data.convert_broken_records_to_nan(df_col, column_number, df_dates, df_periods)
df_col_hour = prepare_data.convert_broken_records_to_nan(df_col_hour, column_number, df_dates_hour, df_periods)

df_analyze = df_col_hour.loc[start_period:end_period]
df_features = df_weer.loc[start_period:end_period]
df_analyze = pd.concat([df_analyze, df_features['solarradiation']], axis=1)

#Impute missing data
df_imputed_hour = prepare_data.replace_broken_records_knn(df_analyze, 0, 1)

#End Data preparation



#4. Analysis

period_freq = 24 #96 for quarterly

#Decomposition analysis
#seasonal_add, non_seasonal_add = analyze_data.analyze_decomp_column(df_imputed_hour, period_freq)
#analyze_data.analyze_decomp_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_decomp_column_period_start_length(df, column_number, start_period, length_period)


#Stationary analysis
    #Determine d-parameter => Which lag is largest (If unclear check differencing ACF)
    #Determine q-parameter => Number of Lags outside of blue zone

    #Full data analysis
#analyze_data.show_plot_acf(df_imputed_hour, column_number)
#analyze_data.show_plot_acf_1_diff(df_imputed_hour, column_number)
#analyze_data.show_plot_acf_2_diff(df_imputed_hour, column_number)

    #Seasonal Component analysis
#analyze_data.show_plot_acf(seasonal_add, column_number)
#analyze_data.show_plot_acf_1_diff(seasonal_add, column_number)
#analyze_data.show_plot_acf_2_diff(seasonal_add, column_number)

    #Non seasonal Component analysis
#analyze_data.show_plot_acf(non_seasonal_add, column_number)
#analyze_data.show_plot_acf_1_diff(non_seasonal_add, column_number)
#analyze_data.show_plot_acf_2_diff(non_seasonal_add, column_number)

    #Double-check d-parameter (For ADF: Under 0.05)
    #Full data analysis
#print('Full data tests:')
#analyze_data.adfuller_test(df_imputed_hour, column_number)
#analyze_data.kpss_test(df_imputed_hour, column_number)

    #Seasonal Component analysis
#print('Seasonal tests:')
#analyze_data.adfuller_test(seasonal_add, column_number)
#analyze_data.kpss_test(seasonal_add, column_number)

    #Non seasonal Component analysis
#print('Non-seasonal tests:')
#analyze_data.adfuller_test(non_seasonal_add, column_number)
#analyze_data.kpss_test(non_seasonal_add, column_number)

    #Determine p-parameter => Which lag is largest (If unclear check next differential PACF)
    #Full data analysis
#analyze_data.show_plot_pacf(df_imputed_hour, column_number)
#analyze_data.show_plot_pacf_1_diff(df_imputed_hour, column_number)
#analyze_data.show_plot_pacf_2_diff(df_imputed_hour, column_number)

    #Seasonal Component analysis
#analyze_data.show_plot_pacf(seasonal_add, column_number)
#analyze_data.show_plot_pacf_1_diff(seasonal_add, column_number)
#analyze_data.show_plot_pacf_2_diff(seasonal_add, column_number)

    #Non seasonal Component analysis
#analyze_data.show_plot_pacf(non_seasonal_add, column_number)
#analyze_data.show_plot_pacf_1_diff(non_seasonal_add, column_number)
#analyze_data.show_plot_pacf_2_diff(non_seasonal_add, column_number)

#End analysis


#5. Model data preparation

#Test samples
df_test_set_2_weeks = df_imputed_hour.loc[start_test_set:end_test_set_2_weeks]
df_test_set_month = df_imputed_hour.loc[start_test_set:end_test_set_month]
df_test_set_2_month = df_imputed_hour.loc[start_test_set:end_test_set_2_month]

df_train, df_test = model_data_preparation.split_data(df_imputed_hour, start_test_set)

#Set these to find test result
training_period = datetime.timedelta(days=26)
start_train_date_1 = pd.to_datetime('2023-06-10 00:00:00+01:00')
split_train_date_1 = start_train_date_1 + training_period

df_train_1, df_validation_1 = model_arima_data.split_data_hour_2_weeks(df_train, start_train_date_1, split_train_date_1)

#End Model data preparation

#6. ARIMA Model

with StepwiseContext(max_steps=100):
    with StepwiseContext(max_dur=200):
        arima_model = auto_arima(df_train_1[df_train_1.columns[0]],start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=24, seasonal=True, error_action='warn', trace=True, suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

print(arima_model.summary())

df_predictions_1 = arima_model.predict(n_periods=len(df_validation_1[df_validation_1.columns[0]]))

print(df_predictions_1)

plt.plot(df_train_1[df_train_1.columns[0]], color = "black")
plt.plot(df_validation_1[df_validation_1.columns[0]], color = "red")
plt.plot(df_predictions_1, color = "blue")
plt.title("Predicted data")
plt.ylabel(df_imputed_hour.columns[column_number])
plt.xlabel('TimeStamp')
plt.show()

#7. Evaluation

rmse = sqrt(mean_squared_error(df_validation_1[df_validation_1.columns[0]], df_predictions_1))
print("RMSE ", rmse)

#End Evaluation

#End main