import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from botPlotService import plot_anomalies, plot_threshold, build_sns_distplot

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from keras import callbacks
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

TIME_STEPS=30
scaler = MinMaxScaler()

def create_df(date, predictions):
    df = pd.DataFrame()
    df['date'] = date 
    df['close'] = predictions
    df['close']=df['close'].astype(float)
    return df

def rename_cols(df):
    df.rename(columns={'close': 'y', 'date': 'ds'},inplace=True)
    df=df.reindex(columns=["ds","y"])
    df['y']=df['y'].astype(float)
    return df

def normalize(df):
    scaled = scaler.fit_transform(df)
    return scaled

def de_normalize(df):
    unscaled = scaler.inverse_transform(df)
    return unscaled

def get_train_set(df):
    train_end = int(len(df) * 0.9)
    train = df.iloc[:train_end]
    return train

def get_test_set(df):
    test_start = int(len(df) * 0.9)
    test = df.iloc[test_start:]
    return test

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

def build_score_df(data, loss, threshold, label, crypto):
    score_df = pd.DataFrame(data)
    score_df['loss'] = loss
    score_df['threshold'] = threshold
    score_df['anomaly'] = score_df['loss'] > score_df['threshold']
    score_df['close'] = data

    #plot_threshold(score_df, label)

    if label == 'Train loss':
        return score_df
        anomalies = score_df.loc[score_df['anomaly'] == True]
        print(anomalies.shape)

        plot_anomalies(score_df, anomalies, crypto)

def find_outliers_with_lstm(X_train, y_train, X_test, y_test, crypto):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X_train.shape[2])))

    model.compile(optimizer='adam', loss='mae')
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)
    model.evaluate(X_test, y_test)

    #train mae loss
    X_train_pred = model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred-X_train), axis=1)
    #build_sns_distplot(train_mae_loss, 'train mae loss')

    threshold = 0.2
    print(f'Reconstruction error threshold: {threshold}')

    #test mae loss
    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)
    #build_sns_distplot(test_mae_loss, 'test mae loss')

    anomalies = build_score_df(y_train, train_mae_loss, threshold, 'Train loss', crypto)
    build_score_df(y_test, test_mae_loss, threshold, 'Test loss', crypto)
    return anomalies
