import datetime as dt
import matplotlib.pyplot as plt

import botApiService 
import botService

print('ETH')

symbol = "ETHUSDT"
interval='1d'

start = str(int(dt.datetime(2021,5,1).timestamp()*1000))
end = str(int(dt.datetime(2023,4,1).timestamp()*1000))

df = botApiService.get_data(symbol, interval, end)

filtered_features_df = df.drop(['open_time', 'high', 'low', 'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore'], axis=1)

scaled_df = botService.normalize_data(filtered_features_df)
train_set = botService.get_train_set(scaled_df)
test_set = botService.get_test_set(scaled_df)

X_train, y_train = botService.create_sequences(train_set[['close']], train_set['close'])
X_test, y_test = botService.create_sequences(test_set[['close']], test_set['close'])

score_df = botService.find_outliers_with_lstm(X_train, y_train, X_test, y_test, symbol)
anomalies = score_df.loc[score_df['anomaly'] == True]
