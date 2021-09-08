# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:22:44 2021

@author: detie
"""

import streamlit as st
from sklearn import preprocessing
import numpy as np

def create_dataset(dataset, n_timesteps):
    dataX, dataY = [], []
    for i in range(len(dataset) - n_timesteps - 1):
        x = dataset[i:i+n_timesteps, 0] # Price between t-n and t-1
        y = dataset[i + n_timesteps, 0] # Price at t
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

def create_conv_dataset(dataset, n_timesteps):
    dataX = []
    dataY = []
    for i in range(len(dataset) - n_timesteps - 1):
        x = dataset[i:i+n_timesteps, :]
        y = dataset[i + n_timesteps, 0]
        dataY.append(y)
        dataX.append(x)
    return np.array(dataX), np.array(dataY)

def create_conv_dataset_rolling(dataset, n_timesteps):
    dataX = []
    dataY = []
    for i in range(len(dataset) - n_timesteps*24 - 1):
        x = dataset[i:i+n_timesteps*24:24, :]       # We take the features at hour i for the n_timesteps previous days
        y = dataset[i + n_timesteps*24, 0]          # We take the target
        dataY.append(y)
        dataX.append(x)
    return np.array(dataX), np.array(dataY)


@st.cache(allow_output_mutation=True)
def preprocessing_univariate(energy):
    # Datas to consider
    energy_price = energy[['price_actual']]
    
    # Separation test/train (75% / 25%)
    energy_price_train = energy_price[:'2017-12-31']
    energy_price_test = energy_price['2018-01-01':]
    
    # scaled
    scaler_price = preprocessing.StandardScaler()
    energy_price_train = scaler_price.fit_transform(energy_price_train)
    energy_price_test = scaler_price.transform(energy_price_test)
    
    # separation X / y
    X_train_ucnn, y_train_ucnn = create_dataset(energy_price_train, 168) # 168 steps = a week
    X_test_ucnn, y_test_ucnn = create_dataset(energy_price_test, 168)
    
    X_train_ucnn = np.expand_dims(X_train_ucnn, 2) # to indicate that each 'individual' as only one feature
    X_test_ucnn = np.expand_dims(X_test_ucnn, 2)
    y_test_ucnn_unscaled = scaler_price.inverse_transform(np.reshape(y_test_ucnn, (-1,1)))
    return X_train_ucnn, y_train_ucnn, X_test_ucnn, y_test_ucnn, y_test_ucnn_unscaled, scaler_price

@st.cache(allow_output_mutation=True)
def preprocessing_multivariate(energy):
    # data to consider
    energy_price_multi = energy[[
        'price_actual',
        'forecast_solar_day_ahead',
        'forecast_wind_onshore_day_ahead',
        'total_load_forecast',
        'day','hour','month'
    ]]
    
    # We add the wind and solar forecast in a new column
    energy_price_multi['forecast_renewable'] = (energy_price_multi.forecast_solar_day_ahead +
                                                energy_price_multi.forecast_wind_onshore_day_ahead)
    energy_price_multi.drop(['forecast_solar_day_ahead','forecast_wind_onshore_day_ahead'],1, inplace = True)
    
    # Also, we can create 2 new features based on forecast_renewable and the total load forecast.
    # Indeed, as previously discussed, the fossil energies will provide the difference between renewable
    # energies and the load (if w simplify the problem), hence it is a direct indicator for the price.
    energy_price_multi['difference'] = (energy_price_multi.forecast_renewable -
                                        energy_price_multi.total_load_forecast)
    energy_price_multi['ratio'] = (energy_price_multi.forecast_renewable /
                                   energy_price_multi.total_load_forecast)
    
    # We want to transform the time parameters so they are made cyclic.
    # Indeed, although there is a proximity in real life between hour 23 and hour 00, the numerical
    # difference would make it difficult for the model to understand it.
    energy_price_multi['hour_sin'] = np.sin(2*np.pi*energy_price_multi.hour / 24)
    energy_price_multi['hour_cos'] = np.cos(2*np.pi*energy_price_multi.hour / 24)
    
    energy_price_multi['month_sin'] = np.sin(2*np.pi*energy_price_multi.month / 12)
    energy_price_multi['month_cos'] = np.cos(2*np.pi*energy_price_multi.month / 12)
    
    energy_price_multi['day_sin'] = np.sin(2*np.pi*energy_price_multi.day / 7)
    energy_price_multi['day_cos'] = np.cos(2*np.pi*energy_price_multi.day / 7)
    
    energy_price_multi.drop(['hour','month', 'day'], 1, inplace = True)
    
    # we need to shift by 24h the first values of forecast wind and solar, since they are prediction
    # as a consequence, we will delete the first 24 lines of the resulting dataframe (nan)
    
    energy_price_multi.forecast_renewable = energy_price_multi.forecast_renewable.shift(24)
    
    energy_price_multi = energy_price_multi['2015-01-02':]
    
    # Separation test/train (75% / 25%)
    energy_price_multi_train = energy_price_multi[:'2017-12-31']
    energy_price_multi_test = energy_price_multi['2018-01-01':]
    
    # Scaled
    standard_scaler = preprocessing.StandardScaler()
    energy_price_multi_train = standard_scaler.fit_transform(energy_price_multi_train)
    energy_price_multi_test = standard_scaler.transform(energy_price_multi_test)
    X_train_mcnn, y_train_mcnn = create_conv_dataset(np.array(energy_price_multi_train), 168)
    X_test_mcnn, y_test_mcnn = create_conv_dataset(np.array(energy_price_multi_test), 168)
    
    # unscaled
    y_test_mcnn_unscaled = np.sqrt(standard_scaler.var_[0])*y_test_mcnn + standard_scaler.mean_[0]
    
    return X_train_mcnn, y_train_mcnn, X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler

@st.cache(allow_output_mutation=True)
def preprocessing_multivariate_rolling(energy):
    # data to consider
    energy_price_multi = energy[[
        'price_actual',
        'forecast_solar_day_ahead',
        'forecast_wind_onshore_day_ahead',
        'total_load_forecast',
        'day','hour','month'
    ]]
    
    # We add the wind and solar forecast in a new column
    energy_price_multi['forecast_renewable'] = (energy_price_multi.forecast_solar_day_ahead +
                                                energy_price_multi.forecast_wind_onshore_day_ahead)
    energy_price_multi.drop(['forecast_solar_day_ahead','forecast_wind_onshore_day_ahead'],1, inplace = True)
    
    # Also, we can create 2 new features based on forecast_renewable and the total load forecast.
    # Indeed, as previously discussed, the fossil energies will provide the difference between renewable
    # energies and the load (if w simplify the problem), hence it is a direct indicator for the price.
    energy_price_multi['difference'] = (energy_price_multi.forecast_renewable -
                                        energy_price_multi.total_load_forecast)
    energy_price_multi['ratio'] = (energy_price_multi.forecast_renewable /
                                   energy_price_multi.total_load_forecast)
    
    # We want to transform the time parameters so they are made cyclic.
    # Indeed, although there is a proximity in real life between hour 23 and hour 00, the numerical
    # difference would make it difficult for the model to understand it.
    energy_price_multi['hour_sin'] = np.sin(2*np.pi*energy_price_multi.hour / 24)
    energy_price_multi['hour_cos'] = np.cos(2*np.pi*energy_price_multi.hour / 24)
    
    energy_price_multi['month_sin'] = np.sin(2*np.pi*energy_price_multi.month / 12)
    energy_price_multi['month_cos'] = np.cos(2*np.pi*energy_price_multi.month / 12)
    
    energy_price_multi['day_sin'] = np.sin(2*np.pi*energy_price_multi.day / 7)
    energy_price_multi['day_cos'] = np.cos(2*np.pi*energy_price_multi.day / 7)
    
    energy_price_multi.drop(['hour','month', 'day'], 1, inplace = True)
    
    # we need to shift by 24h the first values of forecast wind and solar, since they are prediction
    # as a consequence, we will delete the first 24 lines of the resulting dataframe (nan)
    
    energy_price_multi.forecast_renewable = energy_price_multi.forecast_renewable.shift(24)
    
    energy_price_multi = energy_price_multi['2015-01-02':]
    
    # Separation test/train (75% / 25%)
    energy_price_multi_train = energy_price_multi[:'2017-12-31']
    energy_price_multi_test = energy_price_multi['2018-01-01':]
    
    # Scaled
    standard_scaler = preprocessing.StandardScaler()
    energy_price_multi_train = standard_scaler.fit_transform(energy_price_multi_train)
    energy_price_multi_test = standard_scaler.transform(energy_price_multi_test)
    X_train_rolling, y_train_rolling = create_conv_dataset_rolling(np.array(energy_price_multi_train), 168)
    X_test_rolling, y_test_rolling = create_conv_dataset_rolling(np.array(energy_price_multi_test), 168)
    
    # unscaled
    y_test_rolling_unscaled = np.sqrt(standard_scaler.var_[0])*y_test_rolling + standard_scaler.mean_[0]
    
    return X_train_rolling, y_train_rolling, X_test_rolling, y_test_rolling, y_test_rolling_unscaled, standard_scaler