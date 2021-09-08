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


@st.cache(allow_output_mutation=True)
def preprocessing_univariate_load(energy):
    # Datas to consider
    energy_load = energy['total_load_actual']
    
    # Separation test/train (75% / 25%)
    energy_load_train = np.array(energy_load[:'2017-12-31']).reshape((-1,1))
    energy_load_test = np.array(energy_load['2018-01-01':]).reshape((-1,1))
    
    # scaled
    scaler_load = preprocessing.StandardScaler()
    energy_load_train = scaler_load.fit_transform(energy_load_train)
    energy_load_test = scaler_load.transform(energy_load_test)
    
    # separation X / y
    X_train_ucnn_load, y_train_ucnn_load = create_dataset(energy_load_train, 168) # 168 steps = a week
    X_test_ucnn_load, y_test_ucnn_load = create_dataset(energy_load_test, 168)
    
    # to indicate that each 'individual' as only one feature
    X_test_ucnn_load = np.expand_dims(X_test_ucnn_load, 2)
    y_test_ucnn_load = y_test_ucnn_load.reshape((-1,1))
    y_test_ucnn_unscaled_load = scaler_load.inverse_transform(y_test_ucnn_load)
    
    return X_test_ucnn_load, y_test_ucnn_load, y_test_ucnn_unscaled_load, scaler_load

###########################################################################

@st.cache(allow_output_mutation=True)
def preprocessing_multivariate_load(energy):
    # data to consider
    energy_load_multi = energy[['total_load_actual','month','day','hour','temp_mean','pressure','humidity','wind_speed','clouds_all_pop']]

    energy_load_multi['hour_sin'] = np.sin(2*np.pi*energy_load_multi.hour / 24)
    energy_load_multi['hour_cos'] = np.cos(2*np.pi*energy_load_multi.hour / 24)

    energy_load_multi['month_sin'] = np.sin(2*np.pi*energy_load_multi.month / 12)
    energy_load_multi['month_cos'] = np.cos(2*np.pi*energy_load_multi.month / 12)

    energy_load_multi['day_sin'] = np.sin(2*np.pi*energy_load_multi.day / 7)
    energy_load_multi['day_cos'] = np.cos(2*np.pi*energy_load_multi.day / 7)

    energy_load_multi.drop(['hour','month', 'day'], 1, inplace = True)
    
    energy_load_multi_train = energy_load_multi[:'2017-12-31']
    energy_load_multi_test = energy_load_multi['2018-01-01':]
    
    # Scaled
    
    standard_scaler = preprocessing.StandardScaler()
    energy_load_multi_train = standard_scaler.fit_transform(energy_load_multi_train)
    energy_load_multi_test = standard_scaler.transform(energy_load_multi_test)
    X_train_mcnn, y_train_mcnn = create_conv_dataset(np.array(energy_load_multi_train), 168)
    X_test_mcnn, y_test_mcnn = create_conv_dataset(np.array(energy_load_multi_test), 168)
    
    # unscaled
    y_test_mcnn_unscaled = np.sqrt(standard_scaler.var_[0])*y_test_mcnn + standard_scaler.mean_[0]
    
    return X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler