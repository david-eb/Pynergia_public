# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:51:42 2021

@author: detie
"""

import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st

@st.cache
def import_and_clean():
####################### Import raw datas #######################

    energy = pd.read_csv('energy_dataset.csv')
    weather = pd.read_csv('weather_features.csv')
    
####################### delete duplicates of weather #######################
    
    # 24 values per day on 4 years (2015 - 2018) : 35064 total values for df energy.
    # Same goes per city on df weather.
    
    # Values count per city
    weather.city_name.value_counts()
    
    # Deleting duplicates for each city
    weather = weather.drop_duplicates(subset = ['city_name','dt_iso'])
    
####################### Daylight Saving Time #######################
    
    # We are going to delete values in october, and add 'artificial' values in march, in order
    # to deal with Daylight Saving Time
    # New values added will be the same as H+1
    
    # DF ENERGY
    # deleting lines
    energy.drop(energy.index[(energy.time == '2015-10-25 02:00:00+01:00') |
                      (energy.time == '2016-10-30 02:00:00+01:00') |
                      (energy.time == '2017-10-29 02:00:00+01:00') |
                      (energy.time == '2018-10-28 02:00:00+01:00')],
                axis=0,inplace=True)
    
    energy = energy.sort_index().reset_index(drop=True)
    
    # adding rows
    line = energy[28295:28296]
    energy = energy.append(line, ignore_index=False)
    energy = energy.sort_index().reset_index(drop=True)
    energy.iat[28295,0] = '2018-03-25 02:00:00+02:00'
    
    line = energy[19560:19561]
    energy = energy.append(line, ignore_index=False)
    energy = energy.sort_index().reset_index(drop=True)
    energy.iat[19560,0] = '2017-03-26 02:00:00+02:00'
    
    line = energy[10825:10826]
    energy = energy.append(line, ignore_index=False)
    energy = energy.sort_index().reset_index(drop=True)
    energy.iat[10825,0] = '2016-03-27 02:00:00+02:00'
    
    line = energy[2090:2091]
    energy = energy.append(line, ignore_index=False)
    energy = energy.sort_index().reset_index(drop=True)
    energy.iat[2090,0] = '2015-03-29 02:00:00+02:00'
    
    # DF WEATHER
    # deleting lines
    weather.drop(weather.index[(weather.dt_iso == '2015-10-25 02:00:00+01:00') |
                      (weather.dt_iso == '2016-10-30 02:00:00+01:00') |
                      (weather.dt_iso == '2017-10-29 02:00:00+01:00') |
                      (weather.dt_iso == '2018-10-28 02:00:00+01:00')],
                axis=0,inplace=True)
    
    weather = weather.sort_index().reset_index(drop=True)
    weather
    
    # adding rows
    for i in range(4,-1,-1):
        line = weather[28295 + i*35060:28296 + i*35060]
        weather = weather.append(line, ignore_index=False)
        weather = weather.sort_index().reset_index(drop=True)
        weather.iat[28295 + i*35060,0] = '2018-03-25 02:00:00+02:00'
    
        line = weather[19560 + i*35060:19561 + i*35060]
        weather = weather.append(line, ignore_index=False)
        weather = weather.sort_index().reset_index(drop=True)
        weather.iat[19560 + i*35060,0] = '2017-03-26 02:00:00+02:00'
    
        line = weather[10825 + i*35060:10826 + i*35060]
        weather = weather.append(line, ignore_index=False)
        weather = weather.sort_index().reset_index(drop=True)
        weather.iat[10825 + i*35060,0] = '2016-03-27 02:00:00+02:00'
    
        line = weather[2090 + i*35060:2091 + i*35060]
        weather = weather.append(line, ignore_index=False)
        weather = weather.sort_index().reset_index(drop=True)
        weather.iat[2090 + i*35060,0] = '2015-03-29 02:00:00+02:00'
    
####################### NaNs #######################
    
    # Number of NA per column
    #print(energy.isna().sum(axis = 0))
    #print(weather.isna().sum(axis = 0))
    # No NA in weather, some in energy with 2 entire columns empty
    
    # Deleting empty columns
    energy = energy.dropna(axis = 1, how = 'all') 
    
    # Replaceing NA by their median
    energy = energy.fillna(energy.median())
    
    # Deleting columns with only zeros
    energy = energy.drop(['generation fossil coal-derived gas',
                          'generation fossil oil shale',
                          'generation fossil peat',
                          'generation geothermal',
                          'generation marine',
                          'generation wind offshore'], axis = 1)

####################### Deleting uninteresting columns #######################
    
    # In our opinion, the following columns are not worth keeping
    # Regarding the temp, we already have a temp column wich is more interesting. Wind orientation does not seem important.
    # Rain and snow column do not seem reliable (rain_3h sometimes lower than rain_1h : does not make sense)
    # Finally, we have a code for the weather, which sums up weather_main, weather_description and weather_icon.
    weather = weather.drop(['temp_min',
                          'temp_max',
                          'wind_deg',
                          'rain_1h',
                          'rain_3h',
                          'snow_3h',
                          'weather_id',
                          'weather_description',
                          'weather_icon'], axis = 1)
    
####################### Renaming columns #######################
    
    energy = energy.rename(columns = {'generation biomass' : 'biomass',
                                      'generation fossil brown coal/lignite' : 'fossil_brown_coal',
                                      'generation fossil gas' : 'fossil_gas',
                                      'generation fossil hard coal' : 'fossil_hard_coal',
                                      'generation fossil oil' : 'fossil_oil',
                                      'generation hydro pumped storage consumption' : 'hydro_storage_consumption',
                                      'generation hydro run-of-river and poundage' : 'hydro_run_of_river',
                                      'generation hydro water reservoir' : 'hydro_water_reservoir',
                                      'generation nuclear' : 'nuclear',
                                      'generation other' : 'other',
                                      'generation other renewable' : 'other_renewable',
                                      'generation solar' : 'solar',
                                      'generation waste' : 'waste',
                                      'generation wind onshore' : 'wind_onshore',
                                      'forecast solar day ahead' : 'forecast_solar_day_ahead',
                                      'forecast wind onshore day ahead' : 'forecast_wind_onshore_day_ahead',
                                      'total load forecast' : 'total_load_forecast',
                                      'total load actual' : 'total_load_actual',
                                      'price day ahead' : 'price_day_ahead',
                                      'price actual' : 'price_actual'})
    
####################### Extreme values #######################
    
    weather.describe()
    # Pressures and wind speed NOK
    # Replace NOK pressures by patm (=1013 hPa)
    weather[(weather.pressure < 900) | (weather.pressure > 1100)]
    
    # 48 values : replace one by one (some are in Pa, others do not represent anything, ...)
    # For the values without meaning, replace by value of neighbor
    weather.loc[[108572],['pressure']] = 1021
    weather.loc[[108573],['pressure']] = 1021
    weather.loc[[108574],['pressure']] = 1021
    weather.loc[[108575],['pressure']] = 1020
    weather.loc[[108576],['pressure']] = 1019
    weather.loc[[108577],['pressure']] = 1018
    weather.loc[[108578],['pressure']] = 1016
    weather.loc[[108579],['pressure']] = 1016
    weather.loc[[108581],['pressure']] = 1015
    weather.loc[[108582],['pressure']] = 1013
    weather.loc[[108583],['pressure']] = 1013
    weather.loc[[108584],['pressure']] = 1013
    weather.loc[[108585],['pressure']] = 1013
    weather.loc[[108586],['pressure']] = 1012
    weather.loc[[108587],['pressure']] = 1011
    weather.loc[[108588],['pressure']] = 1010
    weather.loc[[108589],['pressure']] = 1008
    weather.loc[[108592],['pressure']] = 1005
    weather.loc[[108593],['pressure']] = 1004
    weather.loc[[108594],['pressure']] = 1003
    weather.loc[[108595],['pressure']] = 1002
    weather.loc[[108596],['pressure']] = 1002
    weather.loc[[108597],['pressure']] = 1002
    weather.loc[[108598],['pressure']] = 1001
    weather.loc[[108599],['pressure']] = 1001
    weather.loc[[108600],['pressure']] = 1000
    weather.loc[[108601],['pressure']] = 999
    weather.loc[[108602],['pressure']] = 999
    weather.loc[[108604],['pressure']] = 999
    weather.loc[[108605],['pressure']] = 999
    weather.loc[[108606],['pressure']] = 1000
    weather.loc[[108609],['pressure']] = 1000
    weather.loc[[108610],['pressure']] = 1000
    weather.loc[[108611],['pressure']] = 1000
    weather.loc[[108612],['pressure']] = 1001
    weather.loc[[108615],['pressure']] = 1001
    weather.loc[[108616],['pressure']] = 1001
    weather.loc[[108617],['pressure']] = 1002
    weather.loc[[108618],['pressure']] = 1002
    weather.loc[[108619],['pressure']] = 1003
    weather.loc[[108620],['pressure']] = 1003
    weather.loc[[108621],['pressure']] = 1003
    weather.loc[[108622],['pressure']] = 1003
    weather.loc[[108623],['pressure']] = 1002
    weather.loc[[108625],['pressure']] = 1002
    weather.loc[[108631],['pressure']] = 1002
    weather.loc[[108632],['pressure']] = 1002
    weather.loc[[108635],['pressure']] = 1002
    weather.loc[[112135],['pressure']] = 1018
    
    # Replace NOK wind speed by median
    weather[(weather.wind_speed > 50)]
    # Only one value seems extreme : 133. This seems to be in km/h instead of m/s.
    weather.loc[[20725],['wind_speed']] = round(133 / 3.6)
    
####################### Index #######################
    
    # We want to have the time as the index
    
    energy.time = energy.time.apply(lambda x: x.split('+')[0])
    energy.time = energy.time.apply(lambda x : dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    energy = energy.set_index('time')
    
    weather.dt_iso = weather.dt_iso.apply(lambda x: x.split('+')[0])
    weather.dt_iso = weather.dt_iso.apply(lambda x : dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    weather = weather.set_index('dt_iso')
    
####################### Creation of useful columns #######################
    
    # Temporal columns
    energy['month'] = energy.index.month
    energy['day'] = energy.index.weekday
    energy['hour'] = energy.index.hour
    energy['hour_a'] = np.arange(0,35064)
    
    weather['month'] = weather.index.month
    weather['day'] = weather.index.weekday
    weather['hour'] = weather.index.hour
    
    # Creation of columns summing fossil energies and renewable energies
    energy['total_renewable'] = (energy.wind_onshore +
                                 energy.solar + 
                                 energy.other_renewable +
                                 energy.biomass +
                                 energy.hydro_run_of_river +
                                 energy.hydro_water_reservoir -
                                 energy.hydro_storage_consumption +
                                 energy.waste)
    
    energy['total_fossil'] = (energy.fossil_brown_coal +
                              energy.fossil_gas +
                              energy.fossil_hard_coal +
                              energy.fossil_oil)
    
    # NB : Maybe could delete individual columns (generation wind onshore, generation solar etc.) later.
    
    # (re)Creation of a column based on weather_main with less modalities :
    
    weather['weather_main'] = weather.weather_main.replace({'clear' : 'good',
                                                            'clouds' : 'average',
                                                            'rain' : 'bad',
                                                            'mist' : 'average',
                                                            'fog' : 'average',
                                                            'drizzle' : 'average',
                                                            'thunderstorm' : 'bad',
                                                            'haze' : 'average',
                                                            'dust' : 'good',
                                                            'snow' : 'bad',
                                                            'smoke' : 'good',
                                                            'squall' : 'average'})
    
####################### Creation of dataframes #######################
    
    # df per city
    madrid_weather = weather[weather.city_name == 'Madrid']
    bilbao_weather = weather[weather.city_name == 'Bilbao']
    seville_weather = weather[weather.city_name == 'Seville']
    barcelona_weather = weather[weather.city_name == ' Barcelona']
    valencia_weather = weather[weather.city_name == 'Valencia']
    
    # Merge with df energy for each city
    madrid = pd.merge(madrid_weather.drop(['month','hour','day'], axis = 1),
                      energy, left_index=True, right_index=True)
    bilbao = pd.merge(bilbao_weather.drop(['month','hour','day'], axis = 1),
                      energy, left_index=True, right_index=True)
    seville = pd.merge(seville_weather.drop(['month','hour','day'], axis = 1),
                       energy, left_index=True, right_index=True)
    barcelona = pd.merge(barcelona_weather.drop(['month','hour','day'], axis = 1),
                         energy, left_index=True, right_index=True)
    valencia = pd.merge(valencia_weather.drop(['month','hour','day'], axis = 1),
                        energy, left_index=True, right_index=True)
    
    # We want to combine energy features with the weather ones.
    # However, the weather is specific to each city.
    # Hence, we set a weight to each city for the weather conditions.
    # Furthermore, there are three coefficients for each city : one is based on the population,
    # the other is based on the wind power installed in the area,
    # and the last one is based on the installed solar power.
    # For further information, please refer to the document XXXX (to add in the github repository).
    
    mad_pop, mad_wind, mad_sol = .3180, .5329, .6287
    bil_pop, bil_wind, bil_sol = .1635, .2204, .0301
    sev_pop, sev_wind, sev_sol = .1939, .1391, .2292
    bar_pop, bar_wind, bar_sol = .1750, .0499, .0352
    val_pop, val_wind, val_sol = .1495, .0577, .0768
    
    energy['temp_mean'] = (madrid_weather.temp * mad_pop +
                           bilbao_weather.temp * bil_pop +
                           seville_weather.temp * sev_pop +
                           barcelona_weather.temp * bar_pop +
                           valencia_weather.temp * val_pop).round(1)
    
    energy['pressure'] = (madrid_weather.pressure * mad_pop +
                           bilbao_weather.pressure * bil_pop +
                           seville_weather.pressure * sev_pop +
                           barcelona_weather.pressure * bar_pop +
                           valencia_weather.pressure * val_pop).round(0)
    
    energy['humidity'] = (madrid_weather.humidity * mad_pop +
                           bilbao_weather.humidity * bil_pop +
                           seville_weather.humidity * sev_pop +
                           barcelona_weather.humidity * bar_pop +
                           valencia_weather.humidity * val_pop).round(0)
    
    energy['wind_speed'] = (madrid_weather.wind_speed * mad_wind +
                           bilbao_weather.wind_speed * bil_wind +
                           seville_weather.wind_speed * sev_wind +
                           barcelona_weather.wind_speed * bar_wind +
                           valencia_weather.wind_speed * val_wind).round(1)
    
    # For clouds cover, we create 2 distinct feature :
    # one with solar coefficient, which can help determine the solar production
    # the other with population coefficient, to help predict the global load
    # (we imagine the cloud cover can influence the load : light necessity etc.)
    energy['clouds_all_sol'] = (madrid_weather.clouds_all * mad_sol +
                           bilbao_weather.clouds_all * bil_sol +
                           seville_weather.clouds_all * sev_sol +
                           barcelona_weather.clouds_all * bar_sol +
                           valencia_weather.clouds_all * val_sol).round(0)
    
    energy['clouds_all_pop'] = (madrid_weather.clouds_all * mad_pop +
                           bilbao_weather.clouds_all * bil_pop +
                           seville_weather.clouds_all * sev_pop +
                           barcelona_weather.clouds_all * bar_pop +
                           valencia_weather.clouds_all * val_pop).round(0)
    
    return energy