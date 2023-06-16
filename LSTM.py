import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

from LSTM_func import *


# Defining some constants for data mining

ALPHA = .8
INTERVAL = '1d'     # Sample rate of historical data
WINDOW_SET_BACK = 100
WINDOW_SIZE = 1000

ALPHA_SMOOTH = .90 # lower = more smooth (0 - 1.0) (.65 default)

st.title("Stock Prediction")

# tuple of stock names to use
# APPL = 12 days, 67%
# SPY = 11 days, 67.31%
stocks = ("AAPL", "GOOG", "SPY", "MSFT")

# create a ui select box for the user
symbol = st.selectbox("Select data set for prediction:", stocks)

# create a ui slider to select the years of prediction
WINDOW_SET_BACK = st.slider("Window Set Back", 1, 200)
n = st.slider("Days of Prediction", 1, 20)
WINDOW = n

# load the data from yahoo finance
# ticker is the name of the stock we want
# st.cache means that even if you reload the page, the data will not have to be reloaded

data_load_state = st.text("Loading stocks data...")
data = load_data(symbol)
data_load_state.text("Loading complete!")
#_____________________________________________________________________________________________________________________________

# smooth out the "spikes" in stock graphs
def _exponential_smooth(data, alpha):
    return data.ewm(alpha=alpha).mean()

tmp_data = _exponential_smooth(data, ALPHA_SMOOTH)
data = data[['Date']].join(tmp_data)

st.text("Smoothing data complete!")

# get the indicators (inputs) for the algo to use
data = get_indicator_data(data)

data_simplfied = data[['Date', 'close']]

# save some testing data 
live_pred_data = data.iloc[-(WINDOW+WINDOW_SIZE+WINDOW_SET_BACK):-(WINDOW+WINDOW_SET_BACK)]

data = produce_truth_vals(data, window=WINDOW)

del (data['close'])
data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here

load_bar = st.text("")
#accuracy, knn_model = cross_Validation(data, load_bar)
knn_model = joblib.load('knn_model_' + symbol +'.pkl')

#accuracy_perc = str(round(accuracy*100, 2))

#st.subheader("KNN Accuracy: " + accuracy_perc + " %")

curr_prices = data_simplfied.iloc[-(WINDOW+WINDOW_SIZE+WINDOW_SET_BACK):-(WINDOW+WINDOW_SET_BACK)][['close', 'Date']]
future_prices = data_simplfied.iloc[-(WINDOW_SIZE+WINDOW_SET_BACK):-(WINDOW_SET_BACK)][['close', 'Date']]

curr_prices['Date'] = pd.to_datetime(curr_prices['Date']).dt.date
future_prices['Date'] = pd.to_datetime(future_prices['Date']).dt.date
curr_prices = curr_prices.reset_index(drop=True)
future_prices = future_prices.reset_index(drop=True)

st.write(curr_prices)
st.write(future_prices)

comparison = (curr_prices['close'] <= future_prices['close']).astype(int)

del(live_pred_data['close'])
live_pred_data = convert_date(live_pred_data)
prediction = knn_model.predict(live_pred_data).astype(int)

comparison = comparison.values
st.text(comparison) 
st.text(prediction) 

count = 0
for i in range(WINDOW_SIZE):
    if comparison[i] == prediction[i]:
        count+=1

st.text(count/WINDOW_SIZE)

msg = ""

if prediction[0] == 1:
    msg = "BUY/KEEP"
else:
    msg = "SELL"

st.header(msg)


