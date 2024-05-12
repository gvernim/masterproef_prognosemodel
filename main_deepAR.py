#Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt
from pandas.tseries.offsets import DateOffset

#Import statistical analysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#Import deepAR model
import mxnet as mx

from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

from tqdm.autonotebook import tqdm

#Import custom classes
import load_data
import visualize_data
import prepare_data
import analyze_data
import model_data_preparation
import model_data_deepar

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
#analyze_data.correlation_between_columns(df_col_hour, df_weer, 0, 4)     #Very strong correlation

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

#Impute missing data
df_imputed_hour = prepare_data.replace_broken_records_knn(df_analyze, 0, 1)

#End Data preparation



#4. Analysis



#End analysis


#5. Model data preparation

#Test samples
#del df_imputed_hour['solarradiation']

df_test_set_1_day = df_imputed_hour.loc[start_test_set:end_test_set_1_day]
df_test_set_2_weeks = df_imputed_hour.loc[start_test_set:end_test_set_2_weeks]
df_test_set_month = df_imputed_hour.loc[start_test_set:end_test_set_month]
df_test_set_2_month = df_imputed_hour.loc[start_test_set:end_test_set_2_month]

df_train, df_test = model_data_preparation.split_data(df_imputed_hour, start_test_set)

#Set length here
context_length=24*31
prediction_length=24*31

#Set required test set here
df_test_set = pd.concat([df_train.iloc[-context_length:], df_test_set_month])

df_train_T = df_train.reset_index()
df_train_T = df_train_T.T
df_test_T = df_test_set.reset_index()
df_test_T = df_test_T.T


target = df_train_T.iloc[1,:]
feat_dynamic = df_train_T.iloc[2:,:]
print(feat_dynamic)

train_ds = ListDataset(
    [
        {
            FieldName.START: df_train.index[0],
            FieldName.TARGET: target,
            FieldName.FEAT_STATIC_CAT: [0],
            FieldName.FEAT_DYNAMIC_REAL: feat_dynamic
        }
    ],
    freq='H'
)

print(train_ds)

target = df_test_T.iloc[1,:]
feat_dynamic = df_test_T.iloc[2:,:]
#feat_dynamic = df_test_T.iloc[2:,-prediction_length:]
print(feat_dynamic)

test_ds = ListDataset(
    [
        {
            FieldName.START: df_test_set.index[0],
            FieldName.TARGET: target,
            FieldName.FEAT_STATIC_CAT: [0],
            FieldName.FEAT_DYNAMIC_REAL: feat_dynamic
        }
    ],
    freq='H'
)

print(test_ds)

#End Model data preparation

#6. DeepAR Model

estimator = DeepAREstimator(freq='H',
                            context_length=context_length,
                            prediction_length=prediction_length,
                            use_feat_dynamic_real=True,
                            num_layers=2,
                            num_cells=40,
                            cell_type='lstm',
                            dropout_rate=0.1,
                            batch_size=128,
                            trainer=Trainer(epochs=10, learning_rate=1e-3)
                            )

predictor = estimator.train(training_data=train_ds, num_workers=4)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)

ts_entry = tss[0]
forecast_entry = forecasts[0]
print(ts_entry)
print(forecast_entry)

#plt.plot(df_test_set_2_weeks)
plt.plot(ts_entry[-24:].to_timestamp(), color='b')
forecast_entry.plot(color='g', intervals=(0.5, 0.9),show_label=True)
ts_entry[0].plot()

plt.show()

#End DeepAR Model

#7. Evaluation

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=1)

print(item_metrics)

#rmse = sqrt(mean_squared_error(df_test_set_2_weeks, forecast))
#mae = mean_absolute_error(df_test_set_2_weeks, forecast)

#print('RMSE: ', str(rmse).replace('.', ','))
#print('MAE: ', str(mae).replace('.', ','))

#End Evaluation

#End Main