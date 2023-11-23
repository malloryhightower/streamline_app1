
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

st.set_page_config(page_title="Location Analysis", page_icon="hi", layout="wide")

data = pd.read_csv('~/Documents/my_app/data/ride_data_location.csv')

# Some number in the range 0-23
st.header("Visualize Pickup Locations")
st.subheader("slide to select time")

hour_to_filter = st.slider('hour', 0, 23, 17)

data['Date/Time'] = pd.to_datetime(data['Date/Time'])
data.rename(columns={'Lat': 'LAT', 'Lon': 'LON'}, inplace=True)

filtered_data = data[data['Date/Time'].dt.hour == hour_to_filter]
number = len(filtered_data)

st.subheader('Map of ' + '{:,}'.format(number)+ ' pickups at %s:00' %hour_to_filter)
st.map(filtered_data)
