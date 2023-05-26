import pandas as pd
import datetime as dt

from binance.spot import Spot as Client

client = Client()

def get_data(symbol, interval, end):
    klines = client.klines(symbol=symbol, interval=interval, limit=1000, endTime=end)
    data = pd.DataFrame(klines)

    # create colums name
    data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']

    # change the timestamp
    data['date'] = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]

    return data