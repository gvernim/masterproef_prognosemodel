#Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

#Import statistical analysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#Import arima model
from pmdarima.arima import auto_arima, ARIMA, StepwiseContext

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

    #Vorm tijdperiodes '2022-01-01 23:00:00+01:00' (Startdate of data)
start_period = pd.to_datetime('2023-01-01 00:00:00+01:00')
end_period = pd.to_datetime('2024-02-29 23:00:00+01:00')
length_period = datetime.timedelta(days=90)

start_test_set = pd.to_datetime('2024-01-01 00:00:00+01:00')
end_test_set_1_day = pd.to_datetime('2024-01-01 23:00:00+01:00')
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

#Test samples
df_weer_test_set_1_day = df_weer.loc[start_test_set:end_test_set_1_day]
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

#visualize_data.visualize_columns(df[start_period:])
#visualize_data.visualize_columns(df_2024)

#analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 0)    #Very weak
#analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 1)    #Weak negative
#analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 2)    #Very weak
#analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 3)    #Weak negative
analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 4)     #Very strong correlation

#visualize_data.visualize_column(df, column_number)
#visualize_data.visualize_column_period_start_end(df, column_number, start_period, end_period)
#visualize_data.visualize_column_period_start_length(df, column_number, start_period, length_period)

#for x in range(len(df.columns)):
#    visualize_data.visualize_column(df, x)

#End visualization data

#3. Data preparation

#Data does not contain NaN values by default for incomplete records
#prepare_data.show_missing_data(df)

#prepare_data.find_outliers_IQR(df_col)

#Adapted per column, 
if column_number==0:
    rolling_records = 32
else:
    rolling_records = 68

#Find faulty data
df_dates, df_periods = prepare_data.find_missing_data_periods(df_col, rolling_records)
df_dates_points = prepare_data.find_missing_data_points(df_col, column_number)

print('The following periods contain possible faulty data: \n', df_periods)

#Add both broken periods and points together
df_dates['broken_record'] = df_dates['broken_record'] | df_dates_points['broken_record']

#Convert found faulty data to 
df_dates_hour = load_data.change_quarterly_index_to_hourly(df_dates)
df_dates_hour['broken_record'] = np.where(df_dates_hour['broken_record'] >= 1, True, False)

#sns.set_theme()
#sns.lineplot(df_dates[df.columns[column_number]])
#sns.lineplot(df_dates['broken_record'], color="red")
#sns.scatterplot(df_dates_points[df.columns[column_number]].where(df_dates['broken_record']), color="red")
#plt.show()

#After check replace with NaN values
df_col = prepare_data.convert_broken_records_to_nan(df_col, column_number, df_dates, df_periods)
df_col_hour = prepare_data.convert_broken_records_to_nan(df_col_hour, column_number, df_dates_hour, df_periods)

df_analyze = df_col_hour.loc[start_period:end_period]
df_features = df_weer.loc[start_period:end_period]
df_analyze = pd.concat([df_analyze, df_features['solarradiation']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['cloudcover']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['windspeed']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['humidity']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['temp']], axis=1)

#Show percentage of missing data
prepare_data.show_missing_data_column(df_analyze, column_number)

#df_period, start, stop = prepare_data.find_largest_period_without_nan(df_col)

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

#print(df_analyze)

df_imputed_hour = prepare_data.replace_broken_records_knn(df_analyze, 0, 1)

#Doesnt work very well
#df_imputed = prepare_data.replace_broken_records_linear_regression(df_analyze)

#For each seperate period
#Equally bad
#for index, row in df_periods.iterrows():
#    if index != 0 and len(df.loc[row.iloc[0]:row.iloc[1]]) < 5000 and index != len(df_periods)-1:
#        first_good_date = df_periods.iloc[index-1, 1]
#        first_good_date_hour = first_good_date.replace(minute=0)
#        print('First: ', first_good_date)
#        print('First hour: ', first_good_date_hour)
#        last_good_date = df_periods.iloc[index+1, 0]
#        last_good_date_hour = last_good_date.replace(minute=0)
#        print('Last: ',last_good_date)
#        print('Last hour: ',last_good_date_hour)

#        if first_good_date_hour > end_period and last_good_date_hour < end_period_2:
#            prepare_data.replace_broken_records_linear_regression(df_analyze.loc[first_good_date_hour:last_good_date_hour])
#        elif first_good_date_hour > end_period and last_good_date_hour > end_period_2:
#            prepare_data.replace_broken_records_linear_regression(df_analyze.loc[first_good_date_hour:end_period_2])

        #df.loc[first_good_date:last_good_date, :] = prepare_data.replace_broken_records_linear_regression(df.loc[first_good_date:last_good_date, :], column_number)

#prepare_data.replace_broken_records_custom(df_col)

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
#        df.loc[first_good_date:last_good_date, :] = prepare_data.replace_broken_records_linear_regression(df.loc[first_good_date:last_good_date, :], column_number)

#End Data preparation

#4. Analysis

period_freq = 24 #96 for quarterly

#Decomposition analysis
#seasonal_add, non_seasonal_add, residuals = analyze_data.analyze_decomp_column(df_imputed_hour, period_freq)
#analyze_data.analyze_decomp_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_decomp_column_period_start_length(df, column_number, start_period, length_period)

#seasonal_add = pd.DataFrame(seasonal_add)
#non_seasonal_add = pd.DataFrame(non_seasonal_add)
#residuals = pd.DataFrame(residuals)

#analyze_data.correlation_between_columns(seasonal_add, df_weer, 0, 0)
#analyze_data.correlation_between_columns(non_seasonal_add, df_weer, 0, 0)
#analyze_data.correlation_between_columns(residuals, df_weer, 0, 0)

#analyze_data.correlation_between_columns(seasonal_add, df_weer, 0, 1)
#analyze_data.correlation_between_columns(non_seasonal_add, df_weer, 0, 1)
#analyze_data.correlation_between_columns(residuals, df_weer, 0, 1)

#analyze_data.correlation_between_columns(seasonal_add, df_weer, 0, 2)
#analyze_data.correlation_between_columns(non_seasonal_add, df_weer, 0, 2)
#analyze_data.correlation_between_columns(residuals, df_weer, 0, 2)

#analyze_data.correlation_between_columns(seasonal_add, df_weer, 0, 3)
#analyze_data.correlation_between_columns(non_seasonal_add, df_weer, 0, 3)
#analyze_data.correlation_between_columns(residuals, df_weer, 0, 3)

#analyze_data.correlation_between_columns(seasonal_add, df_weer, 0, 4)
#analyze_data.correlation_between_columns(non_seasonal_add, df_weer, 0, 4)
#analyze_data.correlation_between_columns(residuals, df_weer, 0, 4)

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

#Extra functions (unused)
#analyze_data.analyze_stat_column(df.iloc[start:stop-1], column_number)
#analyze_data.analyze_stat_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_stat_column_period_start_length(df, column_number, start_period, length_period)


#Autocorrelation analysis (unused)
#analyze_data.analyze_ac_column(df.iloc[start:stop-1], column_number)
#analyze_data.analyze_ac_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_ac_column_period_start_length(df, column_number, start_period, length_period)

#print(df.describe())

#sns.scatterplot(data=df_imputed_hour, x="3VA2 Breaker PV / ActiveEnergyProduction(Productie)", y="solarradiation")
sns.regplot(data=df_imputed_hour, x="3VA2 Breaker PV / ActiveEnergyProduction(Productie)", y="solarradiation", marker="x", line_kws=dict(color="r"))

plt.show()

#End analysis


#5. Model data preparation
#Features verwijderen of toevoegen
#del df_imputed_hour['solarradiation']
#del df_imputed_hour['cloudcover']
del df_imputed_hour['windspeed']
del df_imputed_hour['humidity'] #Not this one
del df_imputed_hour['temp']

#Test samples
df_test_set_1_day = df_imputed_hour.loc[start_test_set:end_test_set_1_day]
df_test_set_2_weeks = df_imputed_hour.loc[start_test_set:end_test_set_2_weeks]
df_test_set_month = df_imputed_hour.loc[start_test_set:end_test_set_month]
df_test_set_2_month = df_imputed_hour.loc[start_test_set:end_test_set_2_month]

df_train, df_test = model_data_preparation.split_data(df_imputed_hour, start_test_set)

#Set test set here
df_test_set = df_test_set_2_weeks

#Change start date df_train
print(len(df_train))
start_date_train = pd.to_datetime('2023-02-01 00:00:00+01:00')
#6 maanden: pd.to_datetime('2023-07-01 00:00:00+01:00')
#4 maanden: pd.to_datetime('2023-09-01 00:00:00+01:00')
#3 maanden: pd.to_datetime('2023-10-01 00:00:00+01:00')
#2 maanden: pd.to_datetime('2023-11-01 00:00:00+01:00')
#1 maand: pd.to_datetime('2023-12-01 00:00:00+01:00')
#2 weken: pd.to_datetime('2023-12-18 00:00:00+01:00')
#1 week: pd.to_datetime('2023-12-25 00:00:00+01:00')
df_train = df_train[start_date_train:] #df_train.columns[0],
print(len(df_train))

#End Model data preparation



#6. ARIMA Model

#Auto arima results
#Standard ARIMA Model
#order=(3,0,2)   #Full training and half
#sorder=(1,0,0)  #Full training
#seasonal_sorder=(2,0,0,24)   #Full training

#Forced d=1
order=(1,1,0)
sorder=(4,1,0)
seasonal_sorder=(1,0,0,24)

#Forced d=2
#order=(0,2,1)
#sorder=(1,2,0)
#seasonal_sorder=(4,0,0,24)

#Forced d=1 and D=1
#order=(3,1,2)
#sorder=(1,1,0)
#seasonal_sorder=(3,1,0,24)

#Forced d=2 and D=1
#order=(3,1,2)
#sorder=(0,1,1)
#seasonal_sorder=(2,0,0,24)

arima_model = ARIMA(order=order)
arima_model = arima_model.fit(df_train[df_train.columns[0]], X=df_train[df_train.columns[1:]])
print('1')
sarima_model = ARIMA(order=sorder, seasonal_order=seasonal_sorder)
sarima_model = sarima_model.fit(df_train[df_train.columns[0]], X=df_train[df_train.columns[1:]])
print('1')

#Rolling Forecast ARIMA model
#1,1,4 AIC=       1,1,5: AIC=     1,2,3: AIC=
#model_fit, predictions = model_arima_data.execute_rolling_arima(train, test, order)

#with StepwiseContext(max_steps=100):
#    with StepwiseContext(max_dur=500):
#        arima_model = auto_arima(df_train[df_train.columns[0]], test='adf', max_p=None, d=1, max_q=None, seasonal=False, information_criterion='aic',error_action='warn', trace=True, suppress_warnings=True, stepwise=True, random_state=20, n_fits=100)
#        sarima_model = auto_arima(df_train[df_train.columns[0]], test='adf', max_p=None, d=1, max_q=None, max_P=None, D=None, max_Q=None, m=24, seasonal=True, error_action='warn', trace=True, suppress_warnings=True, stepwise=True, random_state=20, n_fits=100)

print(arima_model.summary())
print(sarima_model.summary())

df_predictions_1 = arima_model.predict(n_periods=len(df_test_set[df_test_set.columns[0]]), X=df_test_set[df_test_set.columns[1:]])
df_predictions_2 = sarima_model.predict(n_periods=len(df_test_set[df_test_set.columns[0]]), X=df_test_set[df_test_set.columns[1:]])

df_predictions_1.index = df_test_set.index
df_predictions_2.index = df_test_set.index

plt.plot(df_train[df_train.columns[0]], color = "black")
plt.plot(df_test_set[df_test_set.columns[0]], color = "red")
plt.plot(df_predictions_1, color = "blue")
plt.title("Predicted data")
plt.ylabel(df_imputed_hour.columns[column_number])
plt.xlabel('TimeStamp')
plt.show()

plt.plot(df_train[df_train.columns[0]], color = "black")
plt.plot(df_test_set[df_test_set.columns[0]], color = "red")
plt.plot(df_predictions_2, color = "blue")
plt.title("Predicted data")
plt.ylabel(df_imputed_hour.columns[column_number])
plt.xlabel('TimeStamp')
plt.show()

#End ARIMA Model

#7. Evaluation

rmse = sqrt(mean_squared_error(df_test_set[df_test_set.columns[0]], df_predictions_1))
mae = mean_absolute_error(df_test_set[df_test_set.columns[0]], df_predictions_1)
mape = mean_absolute_percentage_error(df_test_set[df_test_set.columns[0]], df_predictions_1)
print('RMSE: ', str(rmse).replace('.', ','))
print('MAE: ', str(mae).replace('.', ','))
print('MAPE: ', str(mape).replace('.', ','))

rmse = sqrt(mean_squared_error(df_test_set[df_test_set.columns[0]], df_predictions_2))
mae = mean_absolute_error(df_test_set[df_test_set.columns[0]], df_predictions_2)
mape = mean_absolute_percentage_error(df_test_set[df_test_set.columns[0]], df_predictions_2)
print('RMSE: ', str(rmse).replace('.', ','))
print('MAE: ', str(mae).replace('.', ','))
print('MAPE: ', str(mape).replace('.', ','))

#End Evaluation

#End main