import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from plotly import graph_objs as go

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric, plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics

from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score 

from keras.models import Sequential
from keras.layers import Activation,Dense,LSTM,Dropout

from botPlotService import plot_predictions
from botStrategy import transform_price_data, determin_buy_event
from botPrepocessingService import de_normalize, create_df, create_df

def build_prophet(df, test_df, interval, frequency, money, symbol):
    df_prophet = Prophet(interval_width=0.95) 
    
    if symbol == "DOGE":
        df_prophet = Prophet(interval_width=0.95, n_changepoints=20, changepoint_prior_scale=0.025) 
    elif symbol == 'ETH':
        df_prophet = Prophet(interval_width=0.95, n_changepoints=40, changepoint_prior_scale=0.2) 
    
    df_prophet.fit(df)

    if frequency == "15m":
        frequency = "15min"
    elif frequency == "1h":
        frequency = "H"
    elif frequency == "12h":
        frequency = "12H"
    elif frequency == "1d":
        frequency = "D"

    df_forecast = df_prophet.make_future_dataframe(periods=interval, freq=frequency)
    df_forecast = df_prophet.predict(df_forecast)

    split = 1000 - interval
    mape_01 = mean_absolute_percentage_error(df['y'], df_forecast.loc[:split-1, ['yhat']]) * 100

    errors = evaluation_tests(test_df['y'], df_forecast.loc[split:, ['yhat']])
    errors["train mape (%)"] = mape_01
    errors = pd.DataFrame(errors, index=[0])
    
    candles = create_df(test_df['ds'], df_forecast.loc[split:, ['yhat']])
    buy_spots, sell_spots, money, total_gain, trading_results = determin_buy_event(candles, interval, money)

    fig = plot_plotly(df_prophet, df_forecast, uncertainty=True, xlabel='date', ylabel='price')
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], name="actual price"))
    if len(buy_spots) > 0 and len(sell_spots) > 0:
        fig.add_trace(go.Scatter(x=buy_spots['date'], y=buy_spots['close'], name="buy spots", mode='markers', connectgaps=False, marker = {'color' : 'darkorange'}))
        fig.add_trace(go.Scatter(x=sell_spots['date'], y=sell_spots['close'], name="sell spots", mode='markers', connectgaps=False, marker = {'color' : 'red'}))
    fig.layout.update(title_text="FB Prophet Model")

    return fig, errors, money, total_gain, trading_results
    #fig = df_prophet.plot(df_forecast, uncertainty=True, xlabel='date', ylabel='price')
    #plt.plot(test_df['ds'], test_df['y'], color="red", zorder=2, label="Actual price")
    #plt.legend()
    #plt.show()
    """fig = df_prophet.plot(df_forecast, uncertainty=True, xlabel='date', ylabel='price')
    plt.plot(test_df['ds'], test_df['y'], color='red', label='actual price')
    plt.legend()
 
    df_cv = cross_validation(df_prophet, initial='720 days', period='180 days' , horizon='100 days')
    #print(df_cv.head())
    df_p = performance_metrics(df_cv)
    print(df_p.head())
    fig1 = plot_cross_validation_metric(df_cv, metric = 'mape')
    #fig1 = df_prophet.plot_components(df_forecast)
    plt.show()"""

def build_lstm(x_train, y_train, x_test, y_test, date_df, split, money):
    np.random.seed(7)
    x_train=np.array(x_train)
    y_train=np.array(y_train)

    model, train_mape = build_cross_val(x_train, y_train)

    x_test=np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 4))

    y_pred = model.predict(x_test)
    y_pred = de_normalize(y_pred)
    y_pred = create_df(date_df[split:], y_pred)
    y_test = create_df(date_df[split:], y_test)

    #errors = evaluation_tests(y_test['close'], y_pred['close'])
    #plot_predictions(y_pred, y_test)

    errors = evaluation_tests(y_test['close'], y_pred['close'])
    errors['train mape (%)'] = train_mape
    errors = pd.DataFrame(errors, index=[0])

    buy_spots, sell_spots, money, total_gain, trading_results = determin_buy_event(y_pred, len(y_pred), money)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test['date'], y=y_test['close'], name="actual price"))
    fig.add_trace(go.Scatter(x=y_pred['date'], y=y_pred['close'], name="predicted price"))
    if len(buy_spots) > 0 and len(sell_spots) > 0:
        fig.add_trace(go.Scatter(x=buy_spots['date'], y=buy_spots['close'], name="buy spots", mode='markers', connectgaps=False, marker = {'color' : 'darkorange'}))
        fig.add_trace(go.Scatter(x=sell_spots['date'], y=sell_spots['close'], name="sell spots", mode='markers', connectgaps=False, marker = {'color' : 'red'}))
    fig.layout.update(title_text="LSTM RNN Model", xaxis_rangeslider_visible=True)

    return fig, errors, money, total_gain, trading_results


def build_cross_val(x, y):
    kfold=KFold(3)
    cvscores = []
    models = []
    losses = []

    for train, test in kfold.split(x, y):
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 4))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, 4))

        neurons=100 
        dropout=0.25
        output_size=1 
        activ_func="tanh"
        loss="mean_squared_error" 
        model = Sequential()
        model.add(LSTM(neurons,return_sequences=True,input_shape=(1,4)))
        model.add(LSTM(90))
    
        model.add(Dropout(dropout))
        
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
        model.compile(loss=loss, optimizer='adam',metrics =["mae","mape"])
        models.append(model)
        history = model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=0,shuffle=False)

        pred = model.predict(X_test)
        pred = de_normalize(pred)
        Y_test = de_normalize(Y_test)
        scores = mean_absolute_percentage_error(Y_test, pred)
        print(scores*100)
        cvscores.append(scores * 100)
        losses.append(history.history['loss'])
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    model = models[cvscores.index(min(cvscores))]
    return model, min(cvscores)

def evaluation_tests(original_df, predict_df):
    errors = {}
    errors["train mape (%)"] = 0
    errors['rmse ($)'] = math.sqrt(mean_squared_error(original_df, predict_df))
    errors['mae ($)'] = mean_absolute_error(original_df, predict_df)
    errors['mape (%)'] = mean_absolute_percentage_error(original_df, predict_df) * 100
    return errors
