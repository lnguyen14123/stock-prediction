import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01" # starting date of jan 1 2015 (for data)
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# tuple of stock names to use
stocks = ("AAPL", "GOOG", "SPY", "MSFT", "Turn this into a textbox so you can input any stock :D")

# create a ui select box for the user
selected_stock = st.selectbox("Select data set for prediction:", stocks)

# create a ui slider to select the years of prediction
n = st.slider("Years of prediction", 1, 4)

period = n*365


# load the data from yahoo finance
# ticker is the name of the stock we want
# st.cache means that even if you reload the page, the data will not have to be reloaded

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # places date the first col of data
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading stocks data...")
data = load_data(selected_stock)
data_load_state.text("Loading complete!")

# give a data preview for the user:)
st.subheader('Raw Data') 
st.write(data.head())

# now lets work on creating a visual graph:
def plot_raw_data():
    fig = go.Figure()
    fig_close = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock opening prices"))
    fig_close.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock closing prices"))
    fig.layout.update(title_text="Open Prices", xaxis_rangeslider_visible=True)
    fig_close.layout.update(title_text="Close Prices", xaxis_rangeslider_visible=True)

    # now plot it!
    st.plotly_chart(fig)
    st.plotly_chart(fig_close)

plot_raw_data()

# now lets do our predictions with the prophet library
df_train = data[['Date', 'Close']]

# prophet requires very specific names for the data frame
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

creating_predictions_text = st.text("Predicting...")

# now lets train the algo
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

creating_predictions_text.text("Predictions Complete!")


st.subheader('Forecast data')
st.write(forecast.tail())

# lets now plot the forecast data
st.write("Forecast Data")
predict_fig = plot_plotly(m, forecast)
st.plotly_chart(predict_fig)

st.write("Forecast Components")
components_fig = m.plot_components(forecast)
st.write(components_fig)


# make an end summary of predictions
st.subheader("End Summary")
summary_disp = forecast.tail(1)
summary_disp = summary_disp.rename(columns={"ds": "Date of Prediction", 
                                    "yhat_lower": "Lower Bound",
                                    "yhat_upper": "Upper Bound",
                                    "yhat": "End Price Prediction"})

#df_train = df_train.drop(['trend', 'trend_lower', 'trend_upper',
#                          'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
#                          'weekly', 'weekly_lower', 'weekly_upper',
#                          'yearly', 'yearly_lower', 'yearly_upper'])

summary_disp = summary_disp[['Date of Prediction', 'End Price Prediction', 'Lower Bound', 'Upper Bound', ]]

st.write("Current Price:")

st.write(data.tail(1))

st.write("Predicted Price:")
st.write(summary_disp)