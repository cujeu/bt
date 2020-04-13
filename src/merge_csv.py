import pandas as pd
import datetime

from config import *
from sp500_symbols import *

def merge_csv(all_tickers, beg_date, end_date):
    all_tickers = ["AAPL","CSCO","AMZN","INTC"]
    # df1 = pd.DataFrame()
    filename = conf_backtest_data_path + all_tickers[0] + '.csv'
    # read csv may create new index column
    df1 = pd.read_csv(filename) #, index_col ='Date') 
    #df1.index.names = ['idx']
    #df1.set_index("idx",inplace=True)
    #slice by date index range
    df1 = df1[(df1.Date >= beg_date) & (df1.Date <= end_date)]
    df1['ticker'] = all_tickers[0]
    for i in range(len(all_tickers)):
        if i > 0:
            filename = conf_backtest_data_path + all_tickers[i] + '.csv'
            df2 = pd.read_csv(filename) 
            #slice by date index range
            df2 = df2[(df2.Date >= beg_date) &
                      (df2.Date <= end_date)]
            df2['ticker'] = all_tickers[i]
            #concat together
            df1 = pd.concat([df1, df2])
    
    """
    remove the first column
    ,Date,Open,High,Low,Close,Adj Close,Volume,ticker
    0,2010-01-04,30.489999771118164,30.64285659790039,30.34000015258789,30.572856903076172,26.538482666015625,123432400,AAPL
    1,2010-01-05,30.65714263916016,30.79857063293457,30.464284896850586,30.625713348388672,26.58436584472656,150476200,AAPL
    """
    #df1 = df1.drop([0], axis=1)
    #df1.set_index(['Date','ticker'], inplace=True)
    filename = conf_backtest_data_path + 'proto.csv'
    # write csv without index column
    df1.to_csv(filename, index=False, encoding='utf8')
                

def merge_sector_csv(sector_name, beg_date, end_date):
    tlist = get_sector_symbols(sector_name)
    merge_csv(tlist, beg_date, end_date)

    
if __name__ == "__main__":
    beg_date = datetime.datetime(2010, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.datetime(2012, 1, 1).strftime('%Y-%m-%d')
    merge_sector_csv("Information Technology",beg_date, end_date)
    ##download_sector_prices("Consumer Staples",10)
    ##download_all_prices(10)

