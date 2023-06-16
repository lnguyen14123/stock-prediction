import streamlit as st
from datetime import date
import joblib


import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score


START = "2015-01-01" # starting date of jan 1 2015 (for data)
TODAY = date.today().strftime("%Y-%m-%d")

"""
Defining some constants for data mining
"""

INTERVAL = '1d'     # Sample rate of historical data
symbol = 'SPY'      # Symbol of the desired stock

# List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']
#INDICATORS = []


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # places date the first col of data
    data.reset_index(inplace=True)
    return data


def produce_truth_vals(data, window):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """
    
    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    
    return data

def get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    data.rename(columns = {'Close': 'close', 'Volume':'volume',
                           'Open': 'open', 'High': 'high',
                           'Low': 'low'}, inplace=True)

    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)

    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features


    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])
    
    return data

def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 22)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    #print(knn_gs.best_params_)
    
    prediction = knn_best.predict(X_test)

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return knn_best

def cross_Validation(data, text_display):

    # Split data into equal partitions of size len_train
    
    num_train = 10 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40 # Length of each train-test set
    
    num_times_run = len(data) // num_train
    load_bar = ("*" * num_times_run)       

    # Lists to store the results from each model
    knn_RESULTS = []

    i = 0
    while True:
        
        start_ind = i * num_train

        #load_bar = load_bar[:i] + "|" + load_bar[i + 1:]
        if len(knn_RESULTS) > 0:
            load_bar = str(i) + " / " + str(num_times_run) + " Current Accuracy: " + str( sum(knn_RESULTS) / len(knn_RESULTS))

        # Partition the data into chunks of size len_train every num_train days

        df = data.iloc[start_ind : (start_ind) + len_train]
        i += 1

        text_display.text(load_bar)

        if len(df) < 40:
            break
        
        y = df['pred']
        X = df.drop('pred', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10,shuffle=False)

        X_train = convert_date(X_train)
        X_test = convert_date(X_test)

        knn_model = _train_KNN(X_train, y_train, X_test, y_test)

        knn_prediction = knn_model.predict(X_test)
        print('knn prediction is ', knn_prediction)
        print('truth values are ', y_test.values)
        
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)

        print(knn_accuracy)
        knn_RESULTS.append(knn_accuracy)
        
    print('KNN Accuracy = ' + str( sum(knn_RESULTS) / len(knn_RESULTS)))

    joblib.dump(knn_model, 'knn_model.pkl')

    return [sum(knn_RESULTS) / len(knn_RESULTS), knn_model]

def convert_date(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].astype(int) / 10**9

    return data

