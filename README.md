# Pynergia

## Executive summary

This project aims at predicting load and prices in the electricity Spanish market for the next 24 hours. The data used were historical data from the Transmission System Operator (TSO) for the year 2015 to 2018 (included).

The project firstly assessed the reliability of day ahead prediction of the TSO (load and prices) with a good score of 1,11%.

The project focus mainly on the use of deep learning algorithms: univariate and multivariate RNN and CNN models, to predict the electricity load and electricity price. The metric used, MAPE (Mean Absolute Percentage Error), reaches 9,24% of electricity load with a multivariate RNN LSTM model and 9,07% for electricity prices with a multivariate CNN model.

Also, we create a SARIMAX model in an uncommon way that reaches a MAPE of less than 6%, but presents some defaults as is.
