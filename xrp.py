from datetime import datetime
from binance.client import Client
import ta
import time
from binance.helpers import round_step_size
import pandas as pd

import sys

import warnings

warnings.filterwarnings("ignore")
print("\n")
print("***" * 12)
print("Robot")
print("***" * 12)

api_key = ""
api_secret = ""

client = Client(api_key, api_secret)

pd.options.display.float_format = '{:,.8f}'.format



def get_tick_size(symbol: str) -> float:
    info = client.futures_exchange_info()

    for symbol_info in info['symbols']:
        if symbol_info['symbol'] == symbol:
            for symbol_filter in symbol_info['filters']:
                if symbol_filter['filterType'] == 'PRICE_FILTER':
                    return float(symbol_filter['tickSize'])


def get_rounded_price(symbol: str, price: float) -> float:
    return round_step_size(price, get_tick_size(symbol))


# Data gathering
def getminutedata(symbol, interval, lookback):
    frame = pd.DataFrame(client.futures_historical_klines(symbol, interval, lookback + " day ago UTC"))
    frame = frame.iloc[:, :6]
    frame.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    frame = frame.set_index("Time")
    frame.index = pd.to_datetime(frame.index, unit="ms")
    frame = frame.astype("float64")
    return frame


def getminutedatabtc(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + " day ago UTC"))
    frame = frame.iloc[:, :6]
    frame.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    #frame = frame.set_index("Time")
    #frame.index = pd.to_datetime(frame.index, unit="ms")
    frame = frame.astype("float64")
    return frame


def technical(df):
    #ohlcv = OHLCVFactory.from_matrix2([df.Open, df.High, df.Low, df.Close, df.Volume, df.Time])
    df["sma_5"] = ta.trend.sma_indicator(df.Close, window=5)
    df["sma_8"] = ta.trend.sma_indicator(df.Close, window=8)
    df["sma_13"] = ta.trend.sma_indicator(df.Close, window=13)
    df["sma_3"] = ta.trend.sma_indicator(df.Close, window=3)
    # df["wma_8"] = ta.trend.wma_indicator(df.Close, window=8)
    # df["wma_13"] = ta.trend.wma_indicator(df.Close, window=13)
    df["Adx"] = ta.trend.adx(df.High, df.Low, df.Close, window=14)
    #adx_values = ADX(14,14,ohlcv)
    #df["Adx"] = ADX(14, 14, ohlcv)
    df["DI+"] = ta.trend.adx_pos(df.High, df.Low, df.Close, window=14)
    df["DI-"] = ta.trend.adx_neg(df.High, df.Low, df.Close, window=14)
    df["ATR"] = ta.volatility.average_true_range(df.High, df.Low, df.Close, window = 14)
    df.dropna(inplace=True)
    return df


# General Parameters

timeframe = sys.argv[3]
lev = int(sys.argv[2])



def Strategy_Short_Open(Ticker, pending_order=False):
    # Leverage

    # Balance information in USDT
    account = client.futures_account_balance()
    balance = float(account[6]["balance"])

    # Import data

    df1 = getminutedata(Ticker + "USDT", timeframe, "2")

    # Apply technicals

    tf1 = technical(df1)

    price = float(tf1.Close.iloc[-1])
    qty = int(balance * lev * 0.90 / price)

    # Adjust leverage
    client.futures_change_leverage(symbol=Ticker + "USDT", leverage=lev)
    if not pending_order:
        print("Short Position:", datetime.utcnow())
        print("Current balance:", balance, "USDT")
        # Send Market Order
        order = client.futures_create_order(
            type='MARKET',

            symbol=Ticker + "USDT",
            side='SELL',
            quantity=qty
        )

        print(order, "\n")

    return True


def Strategy_Short_Close(Ticker, profit, pending_tp=False):
    account = client.futures_account_balance()
    balance = float(account[6]["balance"])

    current_order = []

    df = getminutedata(Ticker + "USDT", timeframe, "2")
    tf = technical(df)

    client.futures_change_leverage(symbol=Ticker + "USDT", leverage=lev)
    # Check for pending order
    last_order = client.futures_get_open_orders(symbol=Ticker + "USDT")

    info = client.futures_position_information(symbol=Ticker + "USDT")  # Position information
    p_amt = float(info[0]["positionAmt"])  # Position amount
    buyprice = info[0]["entryPrice"]  # Position enterance price

    # Take Profit
    if not pending_tp and p_amt != 0 and last_order == []:
        order = client.futures_create_order(
            symbol=Ticker + "USDT",
            type='LIMIT',
            timeInForce="GTC",
            price=get_rounded_price(Ticker + "USDT", float(buyprice) * (1-profit)),
            side='BUY',
            quantity=abs(p_amt)
        )

        print("\nTake profit order has sent:", datetime.utcnow())

        pending_tp = True
        print(order)

    # Stop Loss
    if p_amt != 0:
        if float(tf.sma_3.iloc[-2]) > float(tf.sma_8.iloc[-2]):
            print("Stop Loss Triggered\n")
            order = client.futures_create_order(
                symbol=Ticker + "USDT",
                type='MARKET',
                side='BUY',
                quantity=abs(p_amt),
                reduceOnly=True
            )
            balance = float(account[6]["balance"])
            print("Stop loss filled, current balance:", balance, "USDT\n")
            # Cancel take profit order too since it will cause long position if executed.
            client.futures_cancel_all_open_orders(symbol=Ticker + "USDT")
            print(order)


def Strategy_Long_Open(Ticker, pending_order=False):
    account = client.futures_account_balance()
    balance = float(account[6]["balance"])

    df1 = getminutedata(Ticker + "USDT", timeframe, "2")

    tf1 = technical(df1)

    price = float(tf1.Close.iloc[-1])
    qty = int(balance * lev * 0.90 / price)

    client.futures_change_leverage(symbol=Ticker + "USDT", leverage=lev)
    if not pending_order:
        print("Long Position:", datetime.utcnow())
        print("Current balance:", balance, "USDT")

        order = client.futures_create_order(
            type='MARKET',

            symbol=Ticker + "USDT",
            side='BUY',
            quantity=qty
        )

        print(order, "\n")


def Strategy_Long_Close(Ticker, profit, pending_tp=False):
    account = client.futures_account_balance()
    balance = float(account[6]["balance"])

    current_order = []

    df = getminutedata(Ticker + "USDT", timeframe, "2")
    tf = technical(df)

    client.futures_change_leverage(symbol=Ticker + "USDT", leverage=lev)
    last_order = client.futures_get_open_orders(symbol=Ticker + "USDT")

    info = client.futures_position_information(symbol=Ticker + "USDT")
    p_amt = float(info[0]["positionAmt"])
    buyprice = info[0]["entryPrice"]

    # Take Profit
    if not pending_tp and p_amt != 0 and last_order == []:
        order = client.futures_create_order(
            symbol=Ticker + "USDT",
            type='LIMIT',
            timeInForce="GTC",
            price=get_rounded_price(Ticker + "USDT", float(buyprice) * (1+profit)),
            side='SELL',
            quantity=abs(p_amt)
        )

        print("Take profit order has sent:", datetime.utcnow())

        pending_tp = True

        print(order)

    # Stop Loss
    if p_amt != 0:
        if float(tf.sma_3.iloc[-2]) < float(tf.sma_8.iloc[-2]):
            print("Stop Loss Triggered\n")
            order = client.futures_create_order(
                symbol=Ticker + "USDT",
                type='MARKET',
                side='SELL',
                quantity=abs(p_amt),
                reduceOnly=True
            )
            balance = float(account[6]["balance"])
            print("Stop loss filled, current balance:", balance, "USDT\n")
            print(order)
            client.futures_cancel_all_open_orders(symbol=Ticker + "USDT")

    # if p_amt != 0:
    #    tp_price = get_rounded_price(Ticker + "USDT", float(buyprice) * 0.998)
    #    return tp_price

ticker = sys.argv[1]
account = client.futures_account_balance()
balance = float(account[6]["balance"])
print(ticker)
print("Program successfully opened... ", "\n\nbalance:", balance, "USDT ")

while True:
    try:
        while True:
            
            df_coin = getminutedata(ticker + "USDT", timeframe, "4")
            tf_coin = technical(df_coin)
            info = client.futures_position_information(symbol=ticker + "USDT")
            timenow = datetime.utcnow().minute
            secnow = datetime.utcnow().second

            if timenow % 30 == 0 and secnow < 2:
                print("Checkpoint {}".format(datetime.utcnow()))

            adx_level1 = 30
            adx_level2 = 20
            adx_level3 = 40

            change = abs(tf_coin.Open.iloc[-1] - tf_coin.Close.iloc[-1]) * 100 / tf_coin.Open.iloc[-1]
            

            #long_ma_condition = tf_coin.sma_5.iloc[-1] > tf_coin.sma_8.iloc[-1] and tf_coin.sma_5.iloc[-1] > tf_coin.sma_13.iloc[-1] and tf_coin.sma_3.iloc[-1] > tf_coin.sma_8.iloc[-1]
            #short_ma_condition = tf_coin.sma_5.iloc[-1] < tf_coin.sma_8.iloc[-1] and tf_coin.sma_5.iloc[-1] < tf_coin.sma_13.iloc[-1] and tf_coin.sma_3.iloc[-1] < tf_coin.sma_8.iloc[-1]
            long_ma_condition = tf_coin["DI+"].iloc[-1] > tf_coin["DI-"].iloc[-1] and tf_coin.sma_3.iloc[-1] > \
                                tf_coin.sma_8.iloc[-1]
            short_ma_condition = tf_coin["DI+"].iloc[-1] < tf_coin["DI-"].iloc[-1] and tf_coin.sma_3.iloc[-1] < \
                                tf_coin.sma_8.iloc[-1]
            adx_condition1 = tf_coin.Adx.iloc[-1] > adx_level1 and tf_coin.Adx.iloc[-2] <= adx_level1 and change <= 0.1
            adx_condition2 = tf_coin.Adx.iloc[-1] > adx_level2 and tf_coin.Adx.iloc[-2] <= adx_level2 and change <= 0.1
            adx_condition3 = tf_coin.Adx.iloc[-1] > adx_level3 and tf_coin.Adx.iloc[-2] <= adx_level3 and change <= 0.1

             
            profit = round((tf_coin.ATR.iloc[-1]/tf_coin.Open.iloc[-1]), 4)
            dont_trade_bar_percent = round((tf_coin.ATR.iloc[-1]*100/tf_coin.Open.iloc[-1]), 4)


            percent_2 = abs(float(tf_coin.Close.iloc[-2]) - float(tf_coin.Open.iloc[-2])) * 100 / float(
                tf_coin.Open.iloc[-2])
            percent_3 = abs(float(tf_coin.Close.iloc[-3]) - float(tf_coin.Open.iloc[-3])) * 100 / float(
                tf_coin.Open.iloc[-3])



            # Main Function for Short Position
            if percent_2 < dont_trade_bar_percent and percent_3 < dont_trade_bar_percent and short_ma_condition and (adx_condition1 or adx_condition2 or adx_condition3):
                

                if float(info[0]["positionAmt"]) == 0:
                    Strategy_Short_Open(ticker)
                info = client.futures_position_information(symbol=ticker + "USDT")
                p_amt = (float(info[0]["positionAmt"]))
                


                if p_amt != 0:
                    while True:

                        info = client.futures_position_information(symbol=ticker + "USDT")
                        p_amt1 = (float(info[0]["positionAmt"]))
                        df1 = getminutedata(ticker + "USDT", timeframe, "2")
                        tf1 = technical(df1)
                        price = Strategy_Short_Close(ticker, profit)
                        

                        if float(tf1.sma_3.iloc[-2]) > float(tf1.sma_8.iloc[-2]):

                            break
                        if p_amt1 == 0:
                            account = client.futures_account_balance()
                            balance1 = float(account[6]["balance"])

                            print("\nTake profit filled, current balance:", balance1, "USDT\n")

                            break


            # Main Function for Long Position

            if percent_2 < dont_trade_bar_percent and percent_3 < dont_trade_bar_percent and long_ma_condition and (adx_condition1 or adx_condition2 or adx_condition3):
                

                if float(info[0]["positionAmt"]) == 0:
                    Strategy_Long_Open(ticker)
                info = client.futures_position_information(symbol=ticker + "USDT")

                p_amt = (float(info[0]["positionAmt"]))
                

                if p_amt != 0:
                    while True:

                        info = client.futures_position_information(symbol=ticker + "USDT")
                        p_amt1 = (float(info[0]["positionAmt"]))
                        df1 = getminutedata(ticker + "USDT", timeframe, "2")
                        tf1 = technical(df1)
                        price = Strategy_Long_Close(ticker, profit)
                        
                        if float(tf1.sma_3.iloc[-2]) < float(tf1.sma_8.iloc[-2]):

                            break
                        if p_amt1 == 0:
                            account = client.futures_account_balance()
                            balance1 = float(account[6]["balance"])

                            print("\nTake profit filled, current balance:", balance1, "USDT\n")

                            break
    except Exception as ex:
        print(ex)






