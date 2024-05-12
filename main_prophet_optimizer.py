#Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt, log10
from pandas.tseries.offsets import DateOffset

#Import statistical analysis
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Import prophet model
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
import itertools

#Import custom classes
import load_data
import visualize_data
import prepare_data
import analyze_data
import model_data_preparation
import model_data_prophet

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
end_period = pd.to_datetime('2024-02-29 23:00:00+01:00')

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
df_dates_points = prepare_data.find_missing_data_points(df_col, column_number)

#print('The following periods contain possible faulty data: \n', df_periods)

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
df_analyze = pd.concat([df_analyze, df_features['cloudcover']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['windspeed']], axis=1)
df_analyze = pd.concat([df_analyze, df_features['temp']], axis=1)

#Impute missing data
df_imputed_hour = prepare_data.replace_broken_records_knn(df_analyze, 0, 1)

#End Data preparation



#4. Analysis



#End analysis


#5. Model data preparation

#Test samples

#Lag features not possible in this model
#lag_features = (1, 2, 3)
#df_imputed_hour = model_data_prophet.create_lag_features(df, lag_features)

df_test_set_1_day = df_imputed_hour.loc[start_test_set:end_test_set_1_day]
df_test_set_2_weeks = df_imputed_hour.loc[start_test_set:end_test_set_2_weeks]
df_test_set_month = df_imputed_hour.loc[start_test_set:end_test_set_month]
df_test_set_2_month = df_imputed_hour.loc[start_test_set:end_test_set_2_month]

df_train, df_test = model_data_preparation.split_data(df_imputed_hour, start_test_set)

#Convert to correct format
df_train = model_data_prophet.convert_datetime_index_to_prophet_df(df_train)

#Set required testset here
number_of_predictions = 14*24
df_test_set = model_data_prophet.convert_datetime_index_to_prophet_df(df_test_set_2_weeks)

#End Model data preparation

#6. Prophet Model

#Logarithmic transformation
df_train['y'] = np.log(1 + df_train['y'])
df_train['solarradiation'] = np.log(1 + df_train['solarradiation'])
df_train['cloudcover'] = np.log(1 + df_train['cloudcover'])
df_train['windspeed'] = np.log(1 + df_train['windspeed'])
df_train['temp'] = np.log(1 + df_train['temp'])

df_test_set['solarradiation'] = np.log(1 + df_test_set['solarradiation'])
df_test_set['cloudcover'] = np.log(1 + df_test_set['cloudcover'])
df_test_set['windspeed'] = np.log(1 + df_test_set['windspeed'])
df_test_set['temp'] = np.log(1 + df_test_set['temp'])

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

for params in all_params:
    model = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=True, seasonality_mode='additive', **params)
    #model.add_seasonality(name='daily', period=1, fourier_order=18).add_seasonality(name='yearly', period=365, fourier_order=6) #Set all seasonalities to false

    #Adding feature/regressor
    model.add_regressor('solarradiation')
    #model.add_regressor('cloudcover')
    #model.add_regressor('windspeed')
    #model.add_regressor('temp')

    model.fit(df_train)

    df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='14 days', parallel=None)
    df_p = performance_metrics(df_cv, rolling_window=1)
    print(df_p)
    rmses.append(df_p[['rmse', 'mae']].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

best_params = all_params[np.argmin(rmses)]
print(best_params)

future = model.make_future_dataframe(periods=number_of_predictions, freq='H', include_history=False)

future = future.merge(df_test_set, how='left', on=['ds'])
del future['y']

forecast = model.predict(future)

#Reverse transformation
model.history['y'] = np.exp(model.history['y']) - 1
for col in ['yhat', 'yhat_lower', 'yhat_upper']:#, 'solarradiation', 'cloudcover', 'windspeed', 'temp']:
    forecast[col] = np.exp(forecast[col]) - 1

#fig1 = model.plot(forecast)
#a = add_changepoints_to_plot(fig1.gca(), model, forecast)
#plt.plot('ds', 'y', data=df_test_set, color='red')
#plt.show()

#fig2 = model.plot_components(forecast)
#plt.show()

#fig3 = model.plot(forecast)
#plt.plot('ds', 'y', data=df_test_set, color='red')
#plt.show()

#End Prophet Model

#7. Evaluation

#.iloc[-len(df_test):]  Needed if include_history set to true
rmse = sqrt(mean_squared_error(df_test_set['y'], forecast['yhat']))#.iloc[-len(df_test_set):]))
mae = mean_absolute_error(df_test_set['y'], forecast['yhat'])#.iloc[-len(df_test_set):])

print('RMSE: ', str(rmse).replace('.', ','))
print('MAE: ', str(mae).replace('.', ','))

#End Evaluation

#End Main