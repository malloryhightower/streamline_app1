

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import scipy
import plotly.express as px
import re
from prophet import Prophet
from prophet.utilities import regressor_coefficients
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Forecast", page_icon="hi", layout="wide")
st.sidebar.image("prophet_logo.png",caption="Powered by FB Prophet",use_column_width=True)
#st.sidebar.header("Forecast")

st.markdown("<h1 style='text-align: center; color: black;'>Time Series Modeling</h1>", unsafe_allow_html=True)

## DATA VIZ ##
st.subheader("Data Viz", divider='rainbow')
df = pd.read_csv('data/uber_weather_for_model.csv')
tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.subheader("Total Pickups by Day Over Time")
short = df[df['ds']<'2014-10-01']
short = short.drop(columns=['Unnamed: 0'])
tab1.line_chart(short, x='ds', y='y')
tab2.subheader("raw data")
tab2.write(short)

# regressors #
st.subheader("Regressor Viz", divider='rainbow')
tab1, tab2 = st.tabs(["📈 Temperatures", "📈 Rain"])
tab1.subheader("Min, Max, Temperature by Day (F)")
tab1.line_chart(short, x='ds', y=['Min', 'Max'])
tab2.subheader("Precipitation by Day (in)")
tab2.line_chart(short, x='ds', y='Rain')

## PROPHET MODEL##
st.header("Train and Evaluate Model", divider='rainbow')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.abs((y_true - y_pred) / y_true)) * 100

# st.header("Forecasting Rides per Day")
# st.subheader("data to model, with weather regressors")
# st.write(df)

train_end_dt = '2014-09-30'
m = Prophet(seasonality_mode='additive', weekly_seasonality=True)
m.add_regressor('Max',mode='additive')
m.add_regressor('Min',mode='additive')
m.add_regressor('Rain',mode='additive')
m.add_country_holidays(country_name='US')
m.fit(df)
future = m.make_future_dataframe(periods=92, include_history=True)
future['Max'] = df['Max']
future['Min'] = df['Min']
future['Rain'] = df['Rain']
forecast = m.predict(future)
forecast['y'] = df['y']
train = forecast[forecast['ds']<=train_end_dt]
mape_val = mean_absolute_percentage_error(train['y'], train['yhat'])

st.caption("Train MAPE:")
st.write(round(mape_val,2))
train['ape'] = absolute_percentage_error(train['y'], train['yhat'])

tab1, tab2 = st.tabs(["📈 Error by Day", "📈 Boxplot"])
tab1.subheader("absolute percentage error of the training data ")
tab1.line_chart(train, x='ds', y=['ape'], color=['#ff0899'])
tab2.subheader("Error Distribution")
fig = px.box(train, y="ape", points="all")
tab2.plotly_chart(fig)



st.caption("Components of Forecast")
fig = m.plot_components(forecast)
st.write(fig)

st.header("Forecasted Daily Ride Counts", divider='rainbow')
st.caption('forecast begins in October 2014')
st.line_chart(forecast, x='ds', y=['y', 'yhat'], color=['#4682B4', '#ff0899'])


## User Date inPUT
st.header("Visualize drivers of the forecast", divider='rainbow')
st.caption("understand how the model calculates the prediction")
user_date_input = st.date_input("Select a day", min_value=datetime.date(2014,4,1), max_value=datetime.date(2014,12,31), value=None)
#st.write('Date selected:', user_date_input)

if user_date_input != None:

    ## Waterfall chart
    date = str(user_date_input)
    #date = '2014-05-02'
    sample = forecast.copy()
    sample = sample[sample['ds']==date]

    orig = df[df['ds']==date]
    t_avg= orig['Avg'].values[0]
    r_val = orig['Rain'].values[0]

    if sample['holidays'].values[0] != 0:
        holiday_flag = "yes"
    else:
        holiday_flag = "no"

    if date<=train_end_dt:
        fig = go.Figure(go.Waterfall(
            name = "features", 
            orientation = "v",
            measure = ["relative","relative", "relative", "relative", "total", "total"],
            x = ["trend","weather", "weekly seasonality", "holidays", "predicted rides", "actual rides"],
            textposition = "outside",
            text = [str(round(sample['trend'].values[0])) , str(round(sample['extra_regressors_additive'].values[0])) , str(round(sample['weekly'].values[0]) ), str(round(sample['holidays'].values[0])), str(round(sample['yhat'].values[0])), str(round(sample['y'].values[0]))],
            y = [sample['trend'].values[0], sample['extra_regressors_additive'].values[0] , sample['weekly'].values[0] , sample['holidays'].values[0], sample['yhat'].values[0], sample['y'].values[0]],
            connector = {"visible": False}
        ))
        map_val_here = absolute_percentage_error(sample['y'].values[0], sample['yhat'].values[0])
        st.write('Error on this forecasted day:', round(map_val_here,2))

    else:
        fig = go.Figure(go.Waterfall(
            name = "features", 
            orientation = "v",
            measure = ["relative","relative", "relative", "relative", "total"],
            x = ["trend","weather", "weekly seasonality", "holidays", "predicted rides"],
            textposition = "outside",
            text = [str(round(sample['trend'].values[0])) , str(round(sample['extra_regressors_additive'].values[0])) , str(round(sample['weekly'].values[0]) ), str(round(sample['holidays'].values[0])), str(round(sample['yhat'].values[0]))],
            y = [sample['trend'].values[0] , sample['extra_regressors_additive'].values[0] , sample['weekly'].values[0] , sample['holidays'].values[0], sample['yhat'].values[0] ],
            connector = {"visible": False}
        ))
    fig.update_layout(
            #title ="Impact of Features on Predicted Rides <br><sup>" + "Avg Temp (F) " + str(t_avg)+ " | Rainfall (in) " + str(r_val)+ " | Holiday: " + holiday_flag + "</sup>",
            #title = "Impact of Features on Predicted Rides",
            showlegend = True
    )
    #fig.show()
    st.write('Model Inputs')
    st.markdown("- " + "Avg Temp (F): " + str(t_avg))
    st.markdown("- " + "Rainfall (in): " + str(r_val))
    st.markdown("- " + "Holiday on selected day? " + holiday_flag)
    st.subheader('Impacts of Features on Prediction')
    st.plotly_chart(fig, theme="streamlit")
else:
    st.write('choose a date to see building blocks')




