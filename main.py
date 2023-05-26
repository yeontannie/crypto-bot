import datetime as dt
import streamlit as st
from plotly import graph_objs as go

import botApiService 
import botPrepocessingService
import botModelService

start = str(int(dt.datetime(2021,5,1).timestamp()*1000))
end = str(int(dt.datetime(2023,4,1).timestamp()*1000))

st.title("Crypto prediction magic wisard🧙🏻‍♂️")

symbols = ("BTC", "ETH", "XRP", "DOGE")
selected_symbol = st.selectbox("Select crypto to predict", symbols)

intervals = ("15m", "1h", "12h", "1d")
selected_interval = st.selectbox("Select period of time", intervals)

n_periods = st.slider("Periods of prediction:", 10, 100, step=10)
money_amount = st.number_input("Capital:", min_value=0)

@st.cache_data
def load_data(symbol, interval):
    symbol = f"{symbol}USDT"
    df = botApiService.get_data(symbol, interval, end)
    filtered_features_df = df.drop(['open_time', 'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore'], axis=1)
    return filtered_features_df

data_load_state = st.text("Loading data...")
data = load_data(selected_symbol, selected_interval)
data_load_state.text("Loading data...done!")

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="close price"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

def call_lstm(filtered_features_df, interval, money_amount):
    split = 1000 - interval
    df=filtered_features_df.reindex(columns=["date","open","high","low","volume","close"])
    date_df = df[['date']]
    x=df[["open","high","low","volume"]]
    y=df[["close"]]

    x = botPrepocessingService.normalize(x)
    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_train = botPrepocessingService.normalize(y_train)
    y_test = y[split:]

    fig, errors, money, total_gain, trading_results = botModelService.build_lstm(x_train, y_train, x_test, y_test, date_df, split, money_amount)
    
    st.plotly_chart(fig)
    st.table(trading_results)

    st.subheader("Total")
    st.write("Number of trades: ", str(len(trading_results['Gain'])))
    st.write("Percentage Gains: %.2f%%" % total_gain)
    st.write("Total Profit: $ %.2f" % money)

    st.subheader("Errors (LSTM RNN)")
    st.table(errors)
    

def call_prophet(filtered_features_df, interval, frequency, symbol, money_amount):
    print(symbol)
    split = 1000 - interval
    train_set = filtered_features_df[:split]
    test_set = filtered_features_df[split:]

    train_set = botPrepocessingService.rename_cols(train_set)
    test_set = botPrepocessingService.rename_cols(test_set)

    fig, errors, money, total_gain, trading_results = botModelService.build_prophet(train_set, test_set, interval, frequency, money_amount, symbol)
    
    st.plotly_chart(fig)
    st.table(trading_results)

    st.subheader("Total")
    st.write("Number of trades: ", str(len(trading_results)))
    st.write("Percentage Gains: %.2f%%" % total_gain)
    st.write("Total Profit: $%.2f" % money)

    st.subheader("Errors (FB Prophet)")
    st.table(errors)

build_prophet_state = st.text("Building Prophet Model 📈")
call_prophet(data, n_periods, selected_interval, selected_symbol, money_amount)

build_lstm_state = st.text("Adding some neurons to lstm 🔮")
call_lstm(data, n_periods, money_amount)