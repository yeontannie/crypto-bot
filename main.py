import datetime as dt
import matplotlib.pyplot as plt

import botApiService 
import botService

print('CAKE')

symbol = "CAKEUSDT"
interval='1d'

start = str(int(dt.datetime(2021,5,1).timestamp()*1000))
end = str(int(dt.datetime(2023,4,1).timestamp()*1000))

df = botApiService.get_data(symbol, interval, end)
filtered_features_df = df.drop(['open_time', 'high', 'low', 'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore'], axis=1)
scaled_df = botService.normalize_data(filtered_features_df)

mean = round(scaled_df['close'].mean(),4)
std = round(scaled_df['close'].std(),4)

plt.figtext(.7, .8, f"mean = {mean}")
plt.figtext(.7, .77, f"std = {std}")
scaled_df["close"].plot(title=symbol)
plt.show()
