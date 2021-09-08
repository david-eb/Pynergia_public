# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:48:06 2021

@author: detie
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import statsmodels.api 
import pickle
from joblib import dump, load
from keras.models import load_model
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from preprocessing_data import import_and_clean
from predict_24h_block import predict_24h_uni, predict_24h_multi
from preprocessing_price import create_dataset, create_conv_dataset, create_conv_dataset_rolling
from preprocessing_price import preprocessing_univariate, preprocessing_multivariate, preprocessing_multivariate_rolling

pd.set_option('display.max_columns', None)
sns.set_theme()



#@st.cache(allow_output_mutation=True)
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
 


page = st.sidebar.radio("",
                        options = ['Executive summary',
                                   'Context and Goals of the project',
                                   'Data Presentation & Cleaning',
                                   'Data Viz',
                                   'Modelisation part I : Load',
                                   'Modelisation part II : Price',
                                   'To go further...']
                        )
###########################################################################

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




##############################################################################
##############################################################################
##############################################################################

if page == 'Executive summary':
    img = plt.imread('image_introduction.jpg')
    st.image(img)
    st.markdown('''
                <h1 style="text-align: center;"><span style="color: #333399;"><strong>Pynergia : Load and Price forecast in the Electricity Spanish Market</strong></span></h1>
                <h2 style="text-align: center;"><span style="color: #000000;">From April 2021 to June 2021</span></h2>
                ''', unsafe_allow_html=True)

    st.markdown('''
                **Authors** : Aymeric Longuépée [**(Linkedin)**](https://www.linkedin.com/in/aymeric-longuepee/), Thomas Drizard [**(Linkedin)**](https://www.linkedin.com/in/thomas-drizard-28369345/), David Etien-Berger [**(Linkedin)**](https://www.linkedin.com/in/davidetien/)
                ''')
    
    st.markdown('''
                *Project realized within the Data Scientist training course from* [**Datascientest.com**](https://datascientest.com/)
                ''')
                
    st.header('Executive summary')
    st.markdown('''
                This project aims at predicting load and prices in the electricity Spanish market for the next 24 hours.
                The data used were historical data from the Transmission System Operator (TSO) for
                the year 2015 to 2018 (included).\n
                The project firstly assessed the reliability of day ahead prediction of the TSO 
                (load and prices) with a good score of 1,11%.\n
                The project focus mainly on the use of deep learning algorithms: 
                univariate and multivariate RNN and CNN models, to predict the electricity load and electricity price. 
                The metric used, MAPE 
                (Mean Absolute Percentage Error), reaches 9,24% of electricity load with a 
                multivariate RNN LSTM model and 9,07%  for electricity prices with a multivariate CNN model.\n
                Also, we create a SARIMAX model in an uncommon way that reaches a MAPE of less than 6%, but presents
                some defaults as is.
                ''')
##############################################################################
##############################################################################
##############################################################################

if page == 'Context and Goals of the project':
    st.header('Context and Goals of the project')
    
    st.subheader('Context')
    
    st.markdown('''
                With the emergence of intermittent renewable energy generation and increasing volumes of electricity 
                traded in the markets, forecasting the generation and consumption of electricity, as well as 
                the associated prices, are of increasing importance. This project focuses on the Spanish electricity 
                market at national level, in the context of a country with the most important renewable resources of 
                the European continent, and in 2020 more than 11 GW [[1]](https://www.statista.com/statistics/1003707/installed-solar-pv-capacity-in-spain/) 
                of photovoltaic power and more than 27 GW [[2]](https://renews.biz/66696/spain-increases-wind-capacity-by-172gw-in-2020/) 
                of wind power. The Transmission System Operator (TSO*) is the company in charge of operating the 
                transmission grid (high voltage grid) as well as ensuring the balancing of power and consumption, 
                necessary to avoid black outs and deliver electricity of good quality.                
                ''')
    
    img_diag = plt.imread('diagram_elec.jpg')
    st.image(img_diag)
    
    st.markdown('''
                The spanish TSO (Transmission System Operator), REE (Red Electrica de Espana) makes available generation, consumption and forecast data, 
                through its website [[3]](https://www.ree.es/en/datos/todate)
                or through the website [[4]](https://www.entsoe.eu/data/) 
                of the association of the European TSO (ENTSO-E).
                ''')
                
    img_elec_spain = plt.imread('elec_transmission_spain.jpg')
    st.image(img_elec_spain)
    
    st.subheader('Goals of the Project')
    st.markdown('''
                This project handles the prediction of electricity load and price on the National level of Spain, testing different models.\n
                The target values are load and prices, and we have the “24 hours” prediction of the TSO as well to compare, which will be the reference.\n
                24 hours prediction mean day ahead prediction, different from an “hour by hour approach”.\n
                The data are retrieved from the Spanish TSO platform as well as a weather platform.
                ''')


##############################################################################
##############################################################################
##############################################################################
    
if page == 'Data Presentation & Cleaning':
    st.header('Data Presentation')
    st.markdown('''
                The dataset consists of two files.\n
                The first energy_dataset.csv file contains the following data over 4 years on an hourly basis 
                from the Spanish transmission system operator REE and the association of European 
                transmission system operators ENTSO-E:\n
                1.	Generated power in MW according to the different energy sources 
                (biomass, lignite, coal, natural gas, gas derived from coal, petroleum, 
                 shale oil, peat, geothermal energy, hydraulic pumping station, hydroelectric 
                 in dam and wire of the water, marine, nuclear, other, solar, waste, onshore 
                 and offshore wind power and other renewables)\n
                2.	Forecasted (day ahead) and actual power consumption (“load”)\n
                3.	Forecasted (day ahead) D-1 generation power* from photovoltaic, offshore 
                and onshore wind stations\n
                4.	Forecasted (day ahead) and actual electricity price\n
                The second weather_features.csv file contains the following data from the 
                Open Weather API platform for each of the 5 largest cities in Spain 
                (Madrid, Barcelona, Valencia, Seville, Bilbao), and on an hourly basis over 4 years :\n
                1.	Average, minimum and maximum temperatures (K)\n
                2.	Pressure (hPa)\n
                3.	Humidity (%)\n
                4.	Wind: Wind speed (m / s) and wind direction (angle)\n
                5.	Rain: rain in last hour (mm) and rain in last 3 hours (mm)\n
                6.	Snow: snow in last 3 hours (mm)\n
                7.	Clouds: cloud cover (%)\n
                8.	The description of the weather in full version (weather_description) or 
                abbreviated version (weather_main) as well as a weather code (weather id) as 
                well as a weather icon\n
                The datasets were retrieved from the following links :\n
                [**dataset energy**](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather?select=energy_dataset.csv)\n
                [**dataset weather**](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather?select=weather_features.csv)
                ''')
   
    st.header('Data Cleaning')    
    energy = import_and_clean()
    st.markdown('''
                Preview of the dataset we are going to use, after cleanings.
                ''')
    st.write(energy.head())

##############################################################################
##############################################################################
##############################################################################
    
if page == 'Data Viz':
    st.header('Data Viz')
    energy = import_and_clean()
    st.markdown('''
                Here we are going to explore some features of our dataset through some visualisations.
                ''')

    select_viz = st.selectbox('Which graph would you like to see?',
                          ('Electricity sources in Spain (Pie Chart)',
                           'Total Load per Hour (Barplot)',
                           'Power Generation per Month (Barplot)',
                           'Renewable - Fossil Generation vs Price',
                           'Load and Price distribution')
                         )

##############################################################################
    if select_viz == 'Electricity sources in Spain (Pie Chart)':
        st.subheader('Electricity sources in Spain between 2015 and 2018')
        
        renewable_pilotable = (energy.hydro_water_reservoir.mean() +
                           energy.hydro_run_of_river.mean() +
                           energy.biomass.mean() + energy.other_renewable.mean() + energy.waste.mean() -
                           energy.hydro_storage_consumption.mean())
        renewable_non_pilotable = (energy.solar.mean() + energy.wind_onshore.mean())
        fossil = energy.total_fossil.mean()
        nuclear = energy.nuclear.mean()
        
        fig1 = plt.figure(figsize = (4,6))
        plt.pie([renewable_pilotable, renewable_non_pilotable, fossil, nuclear], 
               labels = ['renewable_pilotable', 'renewable_non_pilotable', 'fossil', 'nuclear'], 
               autopct = lambda x : str(round(x,2))+'%',
               colors = ['#f0ce41','#a7004c','#e10044','#ff5a00'])
        st.pyplot(fig1)
    
##############################################################################    
    if select_viz == 'Total Load per Hour (Barplot)':
        st.subheader('Mean Total Load per Hour')
        st.markdown('''
                    The figure below shows clearly the dependency of the load to the time of the day.
                    As we could have expected, the load is higher in the morning, and in late afternoon, 
                    then decreases through the night.
                    ''')
                    
        tot_p_hour = [energy[energy.hour ==0].total_load_actual.mean(),
                  energy[energy.hour ==1].total_load_actual.mean(),
                  energy[energy.hour ==2].total_load_actual.mean(),
                  energy[energy.hour ==3].total_load_actual.mean(),
                  energy[energy.hour ==4].total_load_actual.mean(),
                  energy[energy.hour ==5].total_load_actual.mean(),
                  energy[energy.hour ==6].total_load_actual.mean(),
                  energy[energy.hour ==7].total_load_actual.mean(),
                  energy[energy.hour ==8].total_load_actual.mean(),
                  energy[energy.hour ==9].total_load_actual.mean(),
                  energy[energy.hour ==10].total_load_actual.mean(),
                  energy[energy.hour ==11].total_load_actual.mean(),
                  energy[energy.hour ==12].total_load_actual.mean(),
                  energy[energy.hour ==13].total_load_actual.mean(),
                  energy[energy.hour ==14].total_load_actual.mean(),
                  energy[energy.hour ==15].total_load_actual.mean(),
                  energy[energy.hour ==16].total_load_actual.mean(),
                  energy[energy.hour ==17].total_load_actual.mean(),
                  energy[energy.hour ==18].total_load_actual.mean(),
                  energy[energy.hour ==19].total_load_actual.mean(),
                  energy[energy.hour ==20].total_load_actual.mean(),
                  energy[energy.hour ==21].total_load_actual.mean(),
                  energy[energy.hour ==22].total_load_actual.mean(),
                  energy[energy.hour ==23].total_load_actual.mean()]            
        fig2 = plt.figure(figsize = (10,5))
        hour = range(24)
        plt.bar(hour, tot_p_hour, color = '#ff5a00', width = 0.9, tick_label = hour)
        plt.xlabel('Hour')
        plt.ylabel('Total Load (MW)')
        st.pyplot(fig2)

##############################################################################
    if select_viz == 'Power Generation per Month (Barplot)':
        st.subheader('Mean power Generation (MW) per Month')
        st.markdown('''
                    This Barplot shows the repartition between the three types of energy throughout the year
                    (mean per month on the 4 years span).\n
                    First, we note that, as could be expected, the peaks in power generation are in winter 
                    (heaters, lights, ...) and in summer (air conditionning, pools, ...).\n
                    Nuclear energy seems pretty steady. Its impact on the forecast of prices and 
                    energy demand is probably minor.\n
                    However, there are important variations between fossil energies and renewable ones. 
                    We can imagine that, because of the lack of reliability of some sources of renewable 
                    energies (wind, solar), the generation cannot be steady ; thus, there is a need to 
                    compensate with the fossil sources.\n
                    The peak of Renewable energies power generation is in march. 
                    More wind in this month? The second graphs below confirms that hypothesis.
                    ''')
         
        renewable_p_month = [energy[energy.month ==1].total_renewable.mean(),
                     energy[energy.month ==2].total_renewable.mean(),
                     energy[energy.month ==3].total_renewable.mean(),
                     energy[energy.month ==4].total_renewable.mean(),
                     energy[energy.month ==5].total_renewable.mean(),
                     energy[energy.month ==6].total_renewable.mean(),
                     energy[energy.month ==7].total_renewable.mean(),
                     energy[energy.month ==8].total_renewable.mean(),
                     energy[energy.month ==9].total_renewable.mean(),
                     energy[energy.month ==10].total_renewable.mean(),
                     energy[energy.month ==11].total_renewable.mean(),
                     energy[energy.month ==12].total_renewable.mean()]

        fossil_p_month = [energy[energy.month ==1].total_fossil.mean(),
                          energy[energy.month ==2].total_fossil.mean(),
                          energy[energy.month ==3].total_fossil.mean(),
                          energy[energy.month ==4].total_fossil.mean(),
                          energy[energy.month ==5].total_fossil.mean(),
                          energy[energy.month ==6].total_fossil.mean(),
                          energy[energy.month ==7].total_fossil.mean(),
                          energy[energy.month ==8].total_fossil.mean(),
                          energy[energy.month ==9].total_fossil.mean(),
                          energy[energy.month ==10].total_fossil.mean(),
                          energy[energy.month ==11].total_fossil.mean(),
                          energy[energy.month ==12].total_fossil.mean()]
                                            
        nuclear_p_month = [energy[energy.month ==1].nuclear.mean(),
                           energy[energy.month ==2].nuclear.mean(),
                           energy[energy.month ==3].nuclear.mean(),
                           energy[energy.month ==4].nuclear.mean(),
                           energy[energy.month ==5].nuclear.mean(),
                           energy[energy.month ==6].nuclear.mean(),
                           energy[energy.month ==7].nuclear.mean(),
                           energy[energy.month ==8].nuclear.mean(),
                           energy[energy.month ==9].nuclear.mean(),
                           energy[energy.month ==10].nuclear.mean(),
                           energy[energy.month ==11].nuclear.mean(),
                           energy[energy.month ==12].nuclear.mean()]
        
        fig3 = plt.figure(figsize=(16, 8))
        month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        plt.bar(range(12), renewable_p_month, color = '#a7004c', width = 0.8, label = 'Renewable', tick_label = month)
        plt.bar(range(12), fossil_p_month, color = '#e10044', width = 0.8, label = 'Fossil',
                bottom = renewable_p_month)
        plt.bar(range(12), nuclear_p_month, color = '#ff5a00', width = 0.8, label = 'Nuclear',
                bottom = (np.array(renewable_p_month) + np.array(fossil_p_month)))
        plt.xlabel('Month')
        plt.ylabel('Mean Power Generation (MW)')
        plt.title('Mean Power Generation (MW) per Month')
        plt.legend();
        st.pyplot(fig3)
        
        windspeed_p_month = [energy[energy.month ==1].wind_speed.mean(),
                   energy[energy.month ==2].wind_speed.mean(),
                   energy[energy.month ==3].wind_speed.mean(),
                   energy[energy.month ==4].wind_speed.mean(),
                   energy[energy.month ==5].wind_speed.mean(),
                   energy[energy.month ==6].wind_speed.mean(),
                   energy[energy.month ==7].wind_speed.mean(),
                   energy[energy.month ==8].wind_speed.mean(),
                   energy[energy.month ==9].wind_speed.mean(),
                   energy[energy.month ==10].wind_speed.mean(),
                   energy[energy.month ==11].wind_speed.mean(),
                   energy[energy.month ==12].wind_speed.mean()]
        
        fig4 = plt.figure(figsize=(16, 8))
        plt.plot(month, windspeed_p_month, c ='y')
        plt.xlabel('Month')
        plt.ylabel('Mean Wind Speed (m/s)')
        plt.title('Mean Wind Speed (m/s) per Month');
        st.pyplot(fig4)
        
##############################################################################
    if select_viz == 'Renewable - Fossil Generation vs Price':
        st.subheader('Renewable and Fossil Generation during the Year vs Price')
        st.markdown('''
                    This lineplot shows clear correlation between the type of energy use
                    and the price.\n
                    Indeed, the higher the use of fossil energy is, the higher the price goes.\n
                    Actually, the price reaches its low in march, matching the highest peak of power generation
                    by renewable energies.\n
                    Hence, there is a positive correlation between fossil generation and prices,
                    and a negative one between renewable energies and prices.
                    ''')
    
        renewable_p_month = [energy[energy.month ==1].total_renewable.mean(),
                         energy[energy.month ==2].total_renewable.mean(),
                         energy[energy.month ==3].total_renewable.mean(),
                         energy[energy.month ==4].total_renewable.mean(),
                         energy[energy.month ==5].total_renewable.mean(),
                         energy[energy.month ==6].total_renewable.mean(),
                         energy[energy.month ==7].total_renewable.mean(),
                         energy[energy.month ==8].total_renewable.mean(),
                         energy[energy.month ==9].total_renewable.mean(),
                         energy[energy.month ==10].total_renewable.mean(),
                         energy[energy.month ==11].total_renewable.mean(),
                         energy[energy.month ==12].total_renewable.mean()]
                    
        fossil_p_month = [energy[energy.month ==1].total_fossil.mean(),
                              energy[energy.month ==2].total_fossil.mean(),
                              energy[energy.month ==3].total_fossil.mean(),
                              energy[energy.month ==4].total_fossil.mean(),
                              energy[energy.month ==5].total_fossil.mean(),
                              energy[energy.month ==6].total_fossil.mean(),
                              energy[energy.month ==7].total_fossil.mean(),
                              energy[energy.month ==8].total_fossil.mean(),
                              energy[energy.month ==9].total_fossil.mean(),
                              energy[energy.month ==10].total_fossil.mean(),
                              energy[energy.month ==11].total_fossil.mean(),
                              energy[energy.month ==12].total_fossil.mean()]
        
        price_mean_p_month = [energy[energy.month ==1]['price_actual'].mean(),
                          energy[energy.month ==2]['price_actual'].mean(),
                          energy[energy.month ==3]['price_actual'].mean(),
                          energy[energy.month ==4]['price_actual'].mean(),
                          energy[energy.month ==5]['price_actual'].mean(),
                          energy[energy.month ==6]['price_actual'].mean(),
                          energy[energy.month ==7]['price_actual'].mean(),
                          energy[energy.month ==8]['price_actual'].mean(),
                          energy[energy.month ==9]['price_actual'].mean(),
                          energy[energy.month ==10]['price_actual'].mean(),
                          energy[energy.month ==11]['price_actual'].mean(),
                          energy[energy.month ==12]['price_actual'].mean()]
        
        month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
        fig5, ax1 = plt.subplots(figsize=(14,6))
        ax1.plot(month, renewable_p_month, color = '#ff5a00', label = 'Renewable')
        ax1.plot(month, fossil_p_month, color = '#f0ce41', label = 'Fossil')
        ax1.set_ylabel('Power Generation (MW)')
        plt.legend(loc = 'lower right')
        ax2 = plt.twinx()
        ax2.plot(month, price_mean_p_month, 'k--', label = 'Price')
        ax2.set_ylabel('Price')
        plt.title('Renewable and Fossil Generation during the year and Price')
        plt.legend();
        st.pyplot(fig5)

##############################################################################
    if select_viz == 'Load and Price distribution':
        st.subheader('Load and Price distribution')
        st.markdown('''
                    Here we want to see the distribution of our two targets, the load and the price.
                    ''')
        
        fig6 = plt.figure(figsize = (5,5))
        sns.violinplot(y = 'total_load_actual', data = energy)
        st.pyplot(fig6)

        fig7 = plt.figure(figsize = (5,5))
        sns.violinplot(y = 'price_actual', data = energy)
        st.pyplot(fig7)
        
##############################################################################
##############################################################################
##############################################################################

if page == 'Modelisation part I : Load':
    st.header('Modelisation part I : Load')
    energy = import_and_clean()
    
    #st.subheader('SARIMAX model')
    #st.markdown('''
                #In this first part, we are going to present the approach we took
                #to build a SARIMAX model.
                #''')
    
    
    st.subheader('Forecast from the dataset')
    st.markdown('''
                 First things first, here is a graph representing the load forecast from the dataset, that
                 we will compared our results to. We do not know, however, how those results were obtained.
                 ''')
    
    energy_comparison_load = energy[['total_load_actual','total_load_forecast']]
    #energy_comparison_load.total_load_forecast = energy_comparison_load.total_load_forecast(24)
    #energy_comparison_load = energy_comparison_load['2015-01-02':]
    
    fig = plt.figure(figsize = (16,8))
    plt.plot(energy_comparison_load.total_load_forecast, color = '#5a1d57', label = 'Predictions')
    plt.plot(energy_comparison_load.total_load_actual, color = '#ff5a00', label = 'True values')
    plt.xlabel('time (hour)')
    plt.ylabel('Load (MW)')
    plt.title('Load vs Time')
    #plt.xlim('2015-01-02','2018-12-31')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
    MAPE_ref = np.mean(np.abs((energy_comparison_load.total_load_actual - energy_comparison_load.total_load_forecast) /
                            energy_comparison_load.total_load_actual)) * 100
    st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_ref)
    
    select_model_load = st.selectbox('Which model would you like to run?',('Load_UCNN','Load_MCNN','Load_URNN','Load_MRNN'))
##############################################################################
    if select_model_load == 'Load_UCNN':
        st.subheader('Univariate CNN model')
        st.markdown('''
                    Here we present the univariate CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])          
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
             # Prep the data
            X_test_ucnn_load, y_test_ucnn_load, y_test_ucnn_unscaled_load, scaler_load = preprocessing_univariate_load(energy)
            # Load model already train
            ucnn_load_model = load_model('ucnn_load_model')
            # Unscale pred and y_true
            pred_ucnn = predict_24h_uni(X_test_ucnn_load, ucnn_load_model)
            pred_ucnn = scaler_load.inverse_transform(np.reshape(pred_ucnn,(-1,1)))
            
            fig_ucnn = plt.figure(figsize = (16,8))
            plt.plot(pred_ucnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_ucnn_unscaled_load[:8568], color = '#ff5a00', label = 'True values')
            #plt.xlim(0,len(pred_ucnn))
            plt.xlabel('time (hour)')
            plt.ylabel('Load (MW)')
            plt.title('Load vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_ucnn)
            
            st.write("MSE : ", mean_squared_error(y_test_ucnn_unscaled_load[:8568], pred_ucnn[:8568]))
            
            MAPE_ucnn = np.mean(np.abs((y_test_ucnn_unscaled_load[:8568] - pred_ucnn[:8568]) / y_test_ucnn_unscaled_load[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_ucnn)
            
########################################################################################
    if select_model_load == 'Load_MCNN':
        
        st.subheader('Multivariate CNN model')
        st.markdown('''
                    Here we present the multivariate CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])          
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        
        if launch == True :
            # Prep the data
            X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler = preprocessing_multivariate_load(energy)
            # Load model already train
            mcnn_model = load_model('mcnn_load_model')
            # Unscale pred and y_true
            pred_mcnn = predict_24h_multi(X_test_mcnn, mcnn_model,168)
            pred_mcnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_mcnn) + standard_scaler.mean_[0]
            
            fig_mcnn = plt.figure(figsize = (16,8))
            plt.plot(pred_mcnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_mcnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            #plt.xlim(0,len(pred_mcnn))
            #plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_mcnn)

            st.write("MSE : ", mean_squared_error(y_test_mcnn_unscaled[:8568], pred_mcnn[:8568]))
            
            MAPE_mcnn = np.mean(np.abs((y_test_mcnn_unscaled[:8568] - pred_mcnn[:8568]) / y_test_mcnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_mcnn)
                     
####################################################################################

    if select_model_load == 'Load_URNN':
        
        st.subheader('Multivariate UNN model')
        st.markdown('''
                    Here we present the univariate RNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences = True, activation = 'relu'),
            tf.keras.layers.LSTM(64, activation = 'relu'),
            tf.keras.layers.Dense(1)
                    ])
    
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        
    
        if launch == True :
            # Prep the data
            X_test_ucnn, y_test_ucnn, y_test_ucnn_unscaled, standard_scaler = preprocessing_univariate_load(energy)
            # Load model already train
            urnn_model = load_model('urnn_load_model')
            # Unscale pred and y_true
            pred_urnn = predict_24h_uni(X_test_ucnn, urnn_model)
            pred_urnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_urnn) + standard_scaler.mean_[0]
            
            fig_urnn = plt.figure(figsize = (16,8))
            plt.plot(pred_urnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_ucnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            #plt.xlim(0,len(pred_urnn))
            #plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('Load (MW)')
            plt.title('Load vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_urnn)

            st.write("MSE : ", mean_squared_error(y_test_ucnn_unscaled[:8568], pred_urnn[:8568]))
            
            MAPE_urnn = np.mean(np.abs((y_test_ucnn_unscaled[:8568] - pred_urnn[:8568]) / y_test_ucnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_urnn)

####################################################################################

    if select_model_load == 'Load_MRNN':
        
        st.subheader('Multivariate UNN model')
        st.markdown('''
                    Here we present the multivariate RNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences = True, activation = 'relu'),
            tf.keras.layers.LSTM(64, activation = 'relu'),
            tf.keras.layers.Dense(1)
                    ])
    
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler = preprocessing_multivariate_load(energy)
            # Load model already train
            mrnn_model = load_model('mrnn_load_model')
            # Unscale pred and y_true
            pred_mrnn = predict_24h_multi(X_test_mcnn, mrnn_model, 168)
            pred_mrnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_mrnn) + standard_scaler.mean_[0]
            
            fig_urnn = plt.figure(figsize = (16,8))
            plt.plot(pred_mrnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_mcnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            #plt.xlim(0,len(pred_mrnn))
            #plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_urnn)

            st.write("MSE : ", mean_squared_error(y_test_mcnn_unscaled[:8568], pred_mrnn[:8568]))
            
            MAPE_mrnn = np.mean(np.abs((y_test_mcnn_unscaled[:8568] - pred_mrnn[:8568]) / y_test_mcnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_mrnn)
            
    
##############################################################################
##############################################################################
##############################################################################
    
if page == 'Modelisation part II : Price':
    st.header('Modelisation part II : Price')
    energy = import_and_clean()

##############################################################################    
    st.markdown('''
                We propose 2 different ways to forecast the price here, and a total of 6 models.\n
                Here are those models :\n
                **Models predicting by block of 24 hours (using 168 timesteps *ie.* a week):**\n
                - Univariate CNN (UCNN)
                - Multivariate CNN (MCNN)
                - Improved MCNN
                - Univariate RNN (URNN)
                - Multivariate RNN (MRNN)\n
                **Models predicting hour by hour 24 hours in advance (so we have at any time a 24 hours rolling forecast):**\n
                - Rolling 24h Improved MCNN
                ''')
                
    st.markdown('''
                *To understand better how we use the features, please go to the forecast process page below*
                ''')


    st.subheader('Forecast from the dataset')
    st.markdown('''
                 First things first, here is a graph representing the forecast from the dataset, that
                 we will compared our results to. We do not know, however, how those results were obtained.
                 ''')
    energy_comparison = energy[['price_day_ahead','price_actual']]
    energy_comparison.price_day_ahead = energy_comparison.price_day_ahead.shift(24)
    energy_comparison = energy_comparison['2015-01-02':]
    
    fig = plt.figure(figsize = (16,8))
    plt.plot(energy_comparison.price_day_ahead, color = '#5a1d57', label = 'Predictions')
    plt.plot(energy_comparison.price_actual, color = '#ff5a00', label = 'True values')
    plt.xlabel('time (hour)')
    plt.ylabel('price')
    plt.title('Price vs Time')
    plt.xlim('2015-01-02','2018-12-31')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
    MAPE_ref = np.mean(np.abs((energy_comparison.price_actual - energy_comparison.price_day_ahead) /
                            energy_comparison.price_actual)) * 100
    st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_ref)

    
    select_model_price = st.selectbox('Which model would you like to run?',
                                      ('UCNN',
                                       'MCNN',
                                       'Improved MCNN',
                                       'URNN',
                                       'MRNN',
                                       'Rolling 24h MCNN',
                                       'Model Comparison & Analysis',
                                       'Forecast process')
                                      )

##############################################################################
    if select_model_price == 'UCNN':
        st.subheader('Univariate CNN model')
        st.markdown('''
                    Here we present the univariate CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])          
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_ucnn, y_train_ucnn, X_test_ucnn, y_test_ucnn, y_test_ucnn_unscaled, scaler_price = preprocessing_univariate(energy)
            # Load model already train
            ucnn_model = load_model('ucnn_model')
            # Unscale pred and y_true
            pred_ucnn = predict_24h_uni(X_test_ucnn, ucnn_model)
            pred_ucnn = scaler_price.inverse_transform(np.reshape(pred_ucnn,(-1,1)))
            
            fig_ucnn = plt.figure(figsize = (16,8))
            plt.plot(pred_ucnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_ucnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_ucnn))
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_ucnn)
            
            st.write("MSE : ", mean_squared_error(y_test_ucnn_unscaled[:8568], pred_ucnn[:8568]))
            
            MAPE_ucnn = np.mean(np.abs((y_test_ucnn_unscaled[:8568] - pred_ucnn[:8568]) / y_test_ucnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_ucnn)
            
##############################################################################
    if select_model_price == 'MCNN':
        st.subheader('Multivariate CNN model')
        st.markdown('''
                    Here we present the multivariate CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])  
            
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_mcnn, y_train_mcnn, X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler = preprocessing_multivariate(energy)
            # Load model already train
            mcnn_model = load_model('mcnn_model')
            # Unscale pred and y_true
            pred_mcnn = predict_24h_multi(X_test_mcnn, mcnn_model,168)
            pred_mcnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_mcnn) + standard_scaler.mean_[0]
            
            fig_mcnn = plt.figure(figsize = (16,8))
            plt.plot(pred_mcnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_mcnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_mcnn))
            plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_mcnn)

            st.write("MSE : ", mean_squared_error(y_test_mcnn_unscaled[:8568], pred_mcnn[:8568]))
            
            MAPE_mcnn = np.mean(np.abs((y_test_mcnn_unscaled[:8568] - pred_mcnn[:8568]) / y_test_mcnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_mcnn)
            
##############################################################################
    if select_model_price == 'Improved MCNN':
        st.subheader('Improved Multivariate CNN model')
        st.markdown('''
                    Here we present an improved version of the previous multivariate CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
                    
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(64, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Dense(1, activation = 'linear')
            ])
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_mcnn, y_train_mcnn, X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler = preprocessing_multivariate(energy)
            # Load model already train
            mcnn_improved_model = load_model('mcnn_improved_model')
            # Unscale pred and y_true
            pred_mcnn_improved = predict_24h_multi(X_test_mcnn, mcnn_improved_model,168)
            pred_mcnn_improved = np.sqrt(standard_scaler.var_[0])*np.array(pred_mcnn_improved) + standard_scaler.mean_[0]
            
            fig_mcnn_improved = plt.figure(figsize = (16,8))
            plt.plot(pred_mcnn_improved[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_mcnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_mcnn_improved))
            plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_mcnn_improved)

            st.write("MSE : ", mean_squared_error(y_test_mcnn_unscaled[:8568], pred_mcnn_improved[:8568]))
            
            MAPE_mcnn_improved = np.mean(np.abs((y_test_mcnn_unscaled[:8568] - pred_mcnn_improved[:8568]) / y_test_mcnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_mcnn_improved)
            
##############################################################################
    if select_model_price == 'URNN':
        st.subheader('Univariate RNN model')
        st.markdown('''
                    Here we present the univariate RNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences = True, activation = 'relu'),
                tf.keras.layers.LSTM(64, activation = 'relu'),
                tf.keras.layers.Dense(1)
            ])
            
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_ucnn, y_train_ucnn, X_test_ucnn, y_test_ucnn, y_test_ucnn_unscaled, standard_scaler = preprocessing_univariate(energy)
            # Load model already train
            urnn_model = load_model('urnn_model')
            # Unscale pred and y_true
            pred_urnn = predict_24h_uni(X_test_ucnn, urnn_model)
            pred_urnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_urnn) + standard_scaler.mean_[0]
            
            fig_urnn = plt.figure(figsize = (16,8))
            plt.plot(pred_urnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_ucnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_urnn))
            plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_urnn)

            st.write("MSE : ", mean_squared_error(y_test_ucnn_unscaled[:8568], pred_urnn[:8568]))
            
            MAPE_urnn = np.mean(np.abs((y_test_ucnn_unscaled[:8568] - pred_urnn[:8568]) / y_test_ucnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_urnn)
            
##############################################################################
    if select_model_price == 'MRNN':
        st.subheader('Multivariate RNN model')
        st.markdown('''
                    Here we present the multivariate RNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences = True, activation = 'relu'),
                tf.keras.layers.LSTM(64, activation = 'relu'),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Dense(1)
            ])
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_mcnn, y_train_mcnn, X_test_mcnn, y_test_mcnn, y_test_mcnn_unscaled, standard_scaler = preprocessing_multivariate(energy)
            # Load model already train
            mrnn_model = load_model('mrnn_model')
            # Unscale pred and y_true
            pred_mrnn = predict_24h_multi(X_test_mcnn, mrnn_model, 168)
            pred_mrnn = np.sqrt(standard_scaler.var_[0])*np.array(pred_mrnn) + standard_scaler.mean_[0]
            
            fig_urnn = plt.figure(figsize = (16,8))
            plt.plot(pred_mrnn[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_mcnn_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_mrnn))
            plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_urnn)

            st.write("MSE : ", mean_squared_error(y_test_mcnn_unscaled[:8568], pred_mrnn[:8568]))
            
            MAPE_mrnn = np.mean(np.abs((y_test_mcnn_unscaled[:8568] - pred_mrnn[:8568]) / y_test_mcnn_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_mrnn)
            
##############################################################################
    if select_model_price == 'Rolling 24h MCNN':
        st.subheader('Rolling 24h MCNN')
        st.markdown('''
                    Here we present the multivariate 24h rolling CNN model we used and the results.\n
                    Model is already fit, but predictions can take awhile.
                    ''')
        
        st.write('''
                 **Model architecture :**
                 ''')
        
        with st.echo():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(32, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.Conv1D(64, kernel_size = 5, activation = 'relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Dense(1, activation = 'linear')
            ])
        
        launch = st.button('Launch model', help = 'Click to launch the model. Can take several minutes.')
        
        if launch == True :
            # Prep the data
            X_train_rolling, y_train_roling, X_test_rolling, y_test_rolling, y_test_rolling_unscaled, standard_scaler = preprocessing_multivariate_rolling(energy)
            # Load model already train
            plus_model = load_model('plus_model')
            # Unscale pred and y_true
            pred_rolling = plus_model.predict(X_test_rolling)
            pred_rolling = np.sqrt(standard_scaler.var_[0])*np.array(pred_rolling) + standard_scaler.mean_[0]
            
            fig_rolling = plt.figure(figsize = (16,8))
            plt.plot(pred_rolling[:8568], color = '#5a1d57', label = 'Predictions')
            plt.plot(y_test_rolling_unscaled[:8568], color = '#ff5a00', label = 'True values')
            plt.xlim(0,len(pred_rolling))
            plt.ylim(0,120)
            plt.xlabel('time (hour)')
            plt.ylabel('price')
            plt.title('Price vs Time')
            plt.legend()
            plt.show()
            st.pyplot(fig_rolling)

            st.write("MSE : ", mean_squared_error(y_test_rolling_unscaled[:8568], pred_rolling[:8568]))
            
            MAPE_rolling = np.mean(np.abs((y_test_rolling_unscaled[:8568] - pred_rolling[:8568]) / y_test_rolling_unscaled[:8568])) * 100
            st.write("Mean Absolute Prediction Error : %0.2f%%"% MAPE_rolling)

##############################################################################                    
    if select_model_price == 'Model Comparison & Analysis':
        st.subheader('Model Comparison & Analysis')
        st.markdown('''
                    The bar plot below presents the performance results for each model tested.\n
                    The metric used is the Mean Absolute Percentage Error (MAPE), which gives a pretty good idea of the performance of the model in our case.\n
                    As expected, multivariate models tend to giver better forecasts, since they have more features, hence more information, to work on.
                    They might need, however, more iterations and/or complex model architecture (*ie.* layers), to give their full potential.
                    RNN models might have better potential than shown on this bar plot, but we had trouble getting consistent results. Furthermore, 
                    they needed very long time to fit ; hence, it resulted dificult to test them.\n
                    The "rolling" model is missing from that plot (MAPE = 13.65%). Even if this model does not seem as efficient as the others, 
                    its working process (hour by hour, 24 hours ahead, forecasting without relying on its own predictions), 
                    makes it a very flexible, reliable and useful model to use, as well as being much quicker to fit.\n
                    Compared to the MAPE score of the forecast from the original dataset, it seems we have better results. However, we do not 
                    know how those forecasts were made (how much time ahead? with which constraints?...), hence it is hard to draw conclusions in this regard.
                    
                    ''')
        img1 = plt.imread('model_comparison.png')
        st.image(img1)


##############################################################################                    
    if select_model_price == 'Forecast process':
        st.subheader('Forecast process')
        st.markdown('**Forecast 24h by 24h, model relying on its own predictions**')
        img1 = plt.imread('24h_forecast.jpg')
        st.image(img1)
        st.markdown('**Forecast rolling 24h : hour by hour 24h ahead**')
        img2 = plt.imread('rolling_24h.jpg')
        st.image(img2)



##############################################################################
##############################################################################
##############################################################################

if page == 'To go further...':
    st.header('To go further...:rocket:')
    st.markdown('''
                - We might think of other parameters to take into acount, 
                or create new features based on the ones we have (sunset and sunrises, holidays, *etc.*).
                - We can imagine we could have better results with more datas, 
                especially regarding the neuronal networks models.
                - We could build rolling 24h model with RNN architecture.
                - Add more epochs, deal with bigger batch_size if more time/computational power.
                ''')
##############################################################################
##############################################################################
##############################################################################