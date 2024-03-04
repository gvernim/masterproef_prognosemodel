#Import standard libraries
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt

#Import statistical analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

#Import arima model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

#Import custom classes
import load_data
import visualize_data
import analyze_data
import model_data

#User variables
filename = 'Merelbeke Energie.csv'
column_number = 0
    #Vorm tijdperiodes '2021-12-31 23:00:00+00:00' (Startdate of data)
start_period = pd.to_datetime('2021-12-31 23:00:00+00:00')
end_period = pd.to_datetime('2022-12-31 23:00:00+00:00')
length_period = datetime.timedelta(days=90)

split_date = pd.to_datetime('2023-05-01 15:00:00+00:00')

#Data headers with number in brackets
    #Timestamp[0] ; PV Productie[1] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[2] ; NA Aankomst / ActiveEnergyProduction(Productie)[3] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[4] ; SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)[5] ;
    #SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[6] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)[7] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[8] ; Verbruik laadpalen(Consumptie)[9] ; TotalProductie[10] ; TotalConsumptie[11] ; UnitOfMeasurement[12] ;

#Start main
#Load and format data
df = load_data.load_data_in_df(filename)

#Data headers with number in brackets after format
    #PV Productie[0] ; NA Aankomst / ActiveEnergyConsumption(Consumptie)[1] ; NA Aankomst / ActiveEnergyProduction(Productie)[2] ;
    #SmartCharging / meter-001 / ActiveEnergyExportTarrif1(Productie)[3] ; SmartCharging / meter-001 / ActiveEnergyExportTarrif2(Productie)[4] ;
    #SmartCharging / meter-001 / ActiveEnergyImportTarrif1(Consumptie)[5] ; SmartCharging / meter-001 / ActiveEnergyImportTarrif2(Consumptie)[6] ;
    #3VA2 Breaker PV / ActiveEnergyConsumption(Consumptie)[7] ; Verbruik laadpalen(Consumptie)[8] ; TotalProductie[9] ; TotalConsumptie[10] ;



#Visualization data

#visualize_data.visualize_columns(df)
#visualize_data.visualize_column(df, column_number)
#visualize_data.visualize_column_period_start_end(df, column_number, start_period, end_period)
#visualize_data.visualize_column_period_start_length(df, column_number, start_period, length_period)

#End visualization data

#Data preparation
#Data does not contain NaN vallues
analyze_data.show_missing_data_1(df)
analyze_data.find_missing_data(df, column_number)

#Analysis

#Decomposition analysis
analyze_data.analyze_decomp_column(df, column_number)
#analyze_data.analyze_decomp_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_decomp_column_period_start_length(df, column_number, start_period, length_period)


#Stationary analysis
analyze_data.show_plot_acf(df, column_number)
#analyze_data.show_plot_acf_1_diff(df, column_number)
#analyze_data.show_plot_acf_2_diff(df, column_number)
analyze_data.adfuller_test(df, column_number)
analyze_data.kpss_test(df, column_number)

#analyze_data.analyze_stat_column(df, column_number)
#analyze_data.analyze_stat_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_stat_column_period_start_length(df, column_number, start_period, length_period)


#Autocorrelation analysis
analyze_data.analyze_ac_column(df, column_number)
#analyze_data.analyze_ac_column_period_start_end(df, column_number, start_period, end_period)
#analyze_data.analyze_ac_column_period_start_length(df, column_number, start_period, length_period)

#print(df.describe())

#End analysis


#Model data preparation

train, test = model_data.split_data(df, column_number, split_date)
#train, test = model_data.split_data_2(df, column_number, 0.33)

#ARIMA Model

model = ARIMA(train, order=(1,1,1))
model.fit(train)
forecast = model.predict(test)

print(forecast)

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(forecast, color = "blue")
plt.title("Train/Test split Data")
plt.ylabel(df.columns[column_number])
plt.xlabel('TimeStamp')
sns.set_theme()
plt.show()

#rmse = sqrt(mean_squared_error(test,forecast))
#print("RMSE: ", rmse)


#End main