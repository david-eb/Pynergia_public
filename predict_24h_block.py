# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:41:18 2021

@author: detie
"""

import numpy as np

def predict_24h_uni(X_test, model):
    '''
    This function is to create prediction based on X_test.
    It will predict the next 24 hours each 24 hour based on the last n_time_steps values.
    It is made for univariate models.
    '''
    pred = []
    for hour in range(0, len(X_test)-24, 24):
        sample = np.reshape(X_test[hour], (1, X_test[0].shape[0], X_test[0].shape[1]))
        for i in range(24):
            next_step = model.predict(sample)
            next_step = np.reshape(next_step, (1,1,1))
            sample = np.concatenate([sample[:,1:,:], next_step], axis = 1)
            pred.append(next_step[0,0,0])
    return pred


def predict_24h_multi(X_test, model, n_time_steps):
    '''
    This function is to create prediction based on X_test.
    It will predict the next 24 hours each 24 hour based on the last n_time_steps values.
    It is made for multivariate models.
    '''
    pred = []
    for hour in range(0, len(X_test)-24, 24):    
        sample = np.reshape(X_test[hour], (1, X_test[0].shape[0], X_test[0].shape[1]))   
        pred_temp_list = []
        for i in range(24):
            pred_temp = model.predict(sample)
            pred_temp = np.reshape(pred_temp, (1,1,1))
            pred.append(pred_temp[0,0,0])
            sample = np.reshape(X_test[hour+i+1], (1, X_test[0].shape[0], X_test[0].shape[1]))
            pred_temp_list.append(pred_temp[0,0,0])
            sample[0,n_time_steps-i-1:n_time_steps,0] = pred_temp_list
    return pred