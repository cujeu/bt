from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import pandas as pd
from datetime import datetime, timedelta
from config import *
from sp500_symbols import *

"""
read csv into dataframe
Date,Open,High,Low,Close,Adj Close,Volume
2010-01-04,30.48,30.64,30.34,30.57,26.53,123432400
"""
def key_fundamental_df(tk, key_list,start_date):
    filename = conf_backtest_data_path + tk + '_key.csv'
    fund_data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
    fund_data_df.index.names = ['date']
    ## set data range by date
    indexDates = fund_data_df[fund_data_df.index < start_date].index
    # Delete these row indexes from dataFrame
    fund_data_df.drop(indexDates , inplace=True)
    #fund_data_df.set_index('Date', inplace=True)
    
    df1 = fund_data_df[key_list]
    return df1

"""
retrieve date not pass cur_date
Date,       marketCap       peRatio   pbRatio  debtToEquity roe
2010-03-27  3.486348e+10   5.403516  0.886029  0.450061     0.163973
"""
def df_date_data(df, tk, col_name, cur_date):
    cell_value = float('nan')
    """
    Get the number of rows: len(df)
    Get the number of columns: len(df.columns)
    Get the number of rows and columns: df.shape
    Get the number of elements: df.size
    for r in range(len(df)):
        dt = df.iloc[r:0]
        if dt > cur_date:
            break
        cell_value = df.iloc[r:2] 
    """
    #access the dataframe using index latest date that precedes cur_date
    #min(df.index)  #max(df[df.index < cur_date.strftime("%Y-%m-%d")].index)
    if len(df[df.index < cur_date].index) > 0:
        cell_value = df.loc[max(df[df.index < cur_date].index)][col_name]
    return cell_value

"""
read csv into dataframe
Date,       marketCap       peRatio   pbRatio  debtToEquity roe
2010-03-27  3.486348e+10   5.403516  0.886029  0.450061     0.163973
2010-06-26  3.362350e+10  10.336153  0.779929  0.501357     0.075456
"""
def hist_price_df(tk, start_date):
    filename = conf_backtest_data_path + tk + '.csv'
    trading_data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
    trading_data_df.drop(['Adj Close'], axis=1, inplace=True)
    trading_data_df.index.names = ['date']
    trading_data_df.rename(columns={'Open' : 'open', 'High' : 'high', 'Low' : 'low',
                                        'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
    ## set data range by date
    indexDates = trading_data_df[trading_data_df.index < start_date].index
    # Delete these row indexes from dataFrame
    trading_data_df.drop(indexDates , inplace=True)
    trading_data_df['openinterest'] = 0
    #trading_data_df.set_index('Date', inplace=True)
    return trading_data_df

"""
merge daily data with quater data
"""
# Use the height and width to calculate the area
#def calculate_area(row):
#    return row['height'] * row['width']
#rectangles_df.apply(calculate_area, axis=1)

def merge_daily_quater_df(daily_df, quater_df, key_list, start_date, end_date):
    for ky in key_list:
        v = 0.0
        s_date = start_date
        # project quater value to rows in daily data frame
        for index, row in quater_df.iterrows():
            q_date = index
            v = quater_df.at[q_date,ky]
            daily_df.loc[(daily_df.index >= s_date) & (daily_df.index < q_date), ky] = v
            s_date = q_date
        if s_date < end_date:
            daily_df.loc[(daily_df.index >= s_date) & (daily_df.index <= end_date), ky] = v
    return daily_df

if __name__ == '__main__':
    start_date = datetime.datetime(2010, 1,1)
    #start_date = datetime.datetime(2015, 1,1)
    end_date = datetime.datetime(2019, 12, 29)
    
    day_df = hist_price_df('AAPL', start_date)
    key_list = ['marketCap','peRatio','pbRatio', 'debtToEquity','roe']
    for k in key_list:
        day_df[k] = 0

    qtr_df=key_fundamental_df('AAPL', key_list ,start_date)
    #print(pdf.head())
    day_df = merge_daily_quater_df(day_df, qtr_df, key_list, start_date, end_date)

