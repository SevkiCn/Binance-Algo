from datetime import datetime
from binance.client import Client
import ta
import time
from binance.helpers import round_step_size
import pandas as pd

import numpy as np


import warnings

warnings.filterwarnings("ignore")



api_key = ""
api_secret = ""
client = Client(api_key, api_secret)

pd.options.display.float_format = '{:,.8f}'.format
account = client.futures_account_balance()
balance = float(account[6]["balance"])
print(balance)


def get_tick_size(symbol: str) -> float:
    info = client.futures_exchange_info()

    for symbol_info in info['symbols']:
        if symbol_info['symbol'] == symbol:
            for symbol_filter in symbol_info['filters']:
                if symbol_filter['filterType'] == 'PRICE_FILTER':
                    return float(symbol_filter['tickSize'])


def get_rounded_price(symbol: str, price: float) -> float:
    return round_step_size(price, get_tick_size(symbol))


def getminutedata(symbol, interval, lookback):
    frame = pd.DataFrame(client.futures_historical_klines(symbol, interval, lookback + " day ago UTC"))
    frame = frame.iloc[:, :6]
    frame.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    frame = frame.set_index("Time")
    frame.index = pd.to_datetime(frame.index, unit="ms")
    frame = frame.astype("float64")
    return frame




def technical(df):
    df["sma_5"] = ta.trend.ema_indicator(df.Close, window=5)
    df["sma_8"] = ta.trend.ema_indicator(df.Close, window=8)
    df["sma_13"] = ta.trend.ema_indicator(df.Close, window=13)
    df["Adx"] = ta.trend.adx(df.High, df.Low, df.Close, window=14)
    df.dropna(inplace=True)
    return df




ticker = "XRPUSDT"

df = getminutedata(ticker, "15m", "45")
tf_coin = technical(df)
tf_coin["Position Info"] = ""


adx_level = 20
adx_level1 = 20
adx_level2 = 40
for i in range(14, len(tf_coin.index)):
    if float(tf_coin.sma_5.iloc[i-1]) < float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-1]) < float(tf_coin.sma_13.iloc[i-1])  and (((float(tf_coin["Adx"].iloc[i-1]) >= adx_level) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level1) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level1)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level2) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level2))):
        tf_coin["Position Info"].iloc[i] = "Short"

    if float(tf_coin.sma_5.iloc[i-1]) < float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-1]) < float(tf_coin.sma_13.iloc[i-1])   and (((float(tf_coin["Adx"].iloc[i-1]) >= adx_level) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level1) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level1)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level2) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level2))):
        tf_coin["Position Info"].iloc[i] = "Short"

    if float(tf_coin.sma_5.iloc[i-1]) > float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-1]) > float(tf_coin.sma_13.iloc[i-1]) and  (((float(tf_coin["Adx"].iloc[i-1]) >= adx_level) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level1) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level1)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level2) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level2))):
        tf_coin["Position Info"].iloc[i] = "Long"

    if float(tf_coin.sma_5.iloc[i-1]) > float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-1]) > float(tf_coin.sma_13.iloc[i-1])  and (((float(tf_coin["Adx"].iloc[i-1]) >= adx_level) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level1) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level1)) or ((float(tf_coin["Adx"].iloc[i-1]) >= adx_level2) and (float(tf_coin["Adx"].iloc[i-2]) <= adx_level2))):
        tf_coin["Position Info"].iloc[i] = "Long"

    if float(tf_coin.sma_5.iloc[i-1]) > float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-2]) < float(tf_coin.sma_8.iloc[i-2]):
        tf_coin["Position Info"].iloc[i] = "Stop_Short"

    if float(tf_coin.sma_5.iloc[i-1]) < float(tf_coin.sma_8.iloc[i-1]) and float(tf_coin.sma_5.iloc[i-2]) > float(tf_coin.sma_8.iloc[i-2]):
        tf_coin["Position Info"].iloc[i] = "Stop_Long"


days = 10
range1 = days*24*4

tf_trade = tf_coin.iloc[-range1:]
tf_trade["Balance"] = ""
tf_trade["Take Profit"] = ""
tf_trade["Position"]=""

balance = 100
lev = 40
tp = 0.002

short_count = 0
profit_short_count = 0
profit_long_count = 0
long_count = 0


for i in range(len(tf_trade.index)):
    if tf_trade["Position Info"].iloc[i] == "Short":
        entry_price = get_rounded_price(ticker, float(tf_trade.Open.iloc[i]))
        take_profit_price = get_rounded_price(ticker, entry_price*(1-tp))
        tf_trade["Take Profit"].iloc[i] = take_profit_price
        print("Short position, entry price:", entry_price, "\nDATE:", tf_trade.index[i])

        short_count += 1


        for j in range(i, len(tf_trade.index)):
            if get_rounded_price(ticker,float(tf_trade.Low.iloc[j])) <= take_profit_price:
                balance = balance*(1+tp*lev) - balance*lev*3/10000
                print("Take profit filled, balance:", balance)
                print("")
                tf_trade["Balance"].iloc[j] = balance
                tf_trade["Position"].iloc[j] = "Take Profit"
                profit_short_count +=1
                i = j

                break
            if tf_trade["Position Info"].iloc[j] == "Stop_Short":
                stop_price = get_rounded_price(ticker, float(tf_trade.Open.iloc[j]))
                loss = ((stop_price-entry_price)/entry_price)*100*lev
                balance = balance*((100-loss)/100) - balance*lev*3/10000
                print("Stop loss filled, balance:", balance)
                print("")
                tf_trade["Balance"].iloc[j] = balance
                i = j
                break
    tf_trade["Balance"].iloc[i] = balance

    if tf_trade["Position Info"].iloc[i] == "Long":
        entry_price = get_rounded_price(ticker, float(tf_trade.Open.iloc[i]))
        take_profit_price = get_rounded_price(ticker, entry_price*(1+tp))
        tf_trade["Take Profit"].iloc[i] = take_profit_price
        print("Long position, entry price:", entry_price, "\nDATE:", tf_trade.index[i])
        long_count += 1

        for j in range(i, len(tf_trade.index)):
            if get_rounded_price(ticker,float(tf_trade.High.iloc[j])) >= take_profit_price:
                balance = balance*(1+tp*lev) - balance*lev*3/10000
                print("Take profit filled, balance:", balance)
                print("")
                tf_trade["Balance"].iloc[j] = balance
                tf_trade["Position"].iloc[j] = "Take Profit"
                i = j
                profit_long_count += 1
                break
            if tf_trade["Position Info"].iloc[j] == "Stop_Long":
                stop_price = get_rounded_price(ticker, float(tf_trade.Open.iloc[j]))
                loss = (abs(stop_price-entry_price)/entry_price)*100*lev
                balance = balance*((100-loss)/100) - balance*lev*3/10000
                print("Stop loss, balance:", balance)
                print("")
                tf_trade["Balance"].iloc[j] = balance
                i = j
                break
    tf_trade["Balance"].iloc[i] = balance

print("***"*10)
print("Backtest Analizi")
print("***"*10)
print("Range: Son %d gün"%days)
print("\nToplam pozisyona giriş:", short_count + long_count)
print("Başarı oranı: %", (profit_long_count+profit_short_count)*100/(short_count+long_count))
print("Kar oranı: %", (balance-100))
print("Başlangıç bakiye: 100 USDT")
print("Mevcut bakiye: ", balance," USDT")
print("Short başarı oranı: %", profit_short_count*100/short_count)
print("Long başarı oranı: %", profit_long_count*100/long_count)

tf_trade.to_excel("backtest.xlsx")