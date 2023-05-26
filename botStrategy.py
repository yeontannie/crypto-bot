import pandas as pd

def transform_price_data(df):
    candles = ['Red']
    for i in range(len(df)):
        if i > 0:
            if df.iloc[i, 1] > df.iloc[i-1, 1]: candles.append('Green')
            else: candles.append('Red')

    df['candles'] = candles
    return df

def take_candle_set(df, m, n):
    return df.iloc[m:n]

def determin_buy_event(df, interval, money):
    m, n = 0, 10
    total_profit = 0
    total_gain = 0

    buy_spots = pd.DataFrame()
    sell_spots = pd.DataFrame()
    trading_results = pd.DataFrame()

    percentage = []
    selling_date = []
    price_bought = []
    price_sold = []
    delta = []

    #candles = transform_price_data(df)
    candle_set = take_candle_set(df, m, n)

    while(n <= interval):
        min_candle = candle_set[candle_set['close'] == min(candle_set['close'])]
        max_candle = candle_set[candle_set['close'] == max(candle_set['close'])]

        if max_candle.index > min_candle.index:
            percent = calc_percentage(min_candle['close'].values[0], max_candle['close'].values[0])
            if percent > 5:
                buy_spots = buy_spots.append(min_candle.iloc[:, :2], ignore_index = True)
                sell_spots = sell_spots.append(max_candle.iloc[:, :2], ignore_index = True)
                total_profit += ((money / min_candle.iloc[0, 1]) * max_candle.iloc[0, 1]) - money
                total_gain += percent
                m = n 
                
                percentage.append(("%.2f%%" % percent))
                selling_date.append(str(max_candle['date'].values[0])[:10])
                price_bought.append(("$%.2f" % min_candle['close'].values[0]))
                price_sold.append(("$%.2f" % max_candle['close'].values[0]))
                delta.append(("$%.2f" % (max_candle['close'].values[0] - min_candle['close'].values[0])))
        n = n + 10
        if n <= 100:    
            candle_set = take_candle_set(df, m, n)
            candle_set = candle_set[candle_set['date'] > min_candle['date'].values[0]]
    
    trading_results['Date of sell'] = selling_date
    trading_results['Price bought'] = price_bought
    trading_results['Price sold'] = price_sold
    trading_results['Delta'] = delta
    trading_results['Gain'] = percentage

    return buy_spots, sell_spots, total_profit, total_gain, trading_results
        
def calc_percentage(prev_close, current_close):
    return ((current_close - prev_close)/prev_close)*100