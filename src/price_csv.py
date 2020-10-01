import pandas as pd
from yahoofinancials import YahooFinancials
import datetime

from config import *
from sp500_symbols import *

def get_price_yahoo(all_tickers, years):
    #,"CSCO","AMZN","INTC"]
    # extracting stock data (historical close price) for the stocks identified
    close_prices = pd.DataFrame()
    end_date = (datetime.date.today()-datetime.timedelta(1)).strftime('%Y-%m-%d')
    beg_date = datetime.datetime(2010, 1, 1).strftime('%Y-%m-%d')
    #beg_date = (datetime.date.today()-datetime.timedelta(365*years)).strftime('%Y-%m-%d')
    cp_tickers = all_tickers
    attempt = 0
    drop = []
    # names in cols are features of get_historical_price_data
    cols = ["formatted_date","open", "high", "low", "close","adjclose","volume"]
    #cols = ["Date","Open", "High", "Low", "Close","Adj Close","Volume"]
    # other order: Date,Open,High,Low,Close,Adj Close,Volume
    # anther order Date,Open,High,Low,Close,Volume,Adj Close
    while len(cp_tickers) != 0 and attempt <= 3:
        print("-----------------")
        print("attempt number ",attempt)
        print("-----------------")
        cp_tickers = [j for j in cp_tickers if j not in drop]
        for i in range(len(cp_tickers)):
            try:
                yahoo_financials = YahooFinancials(cp_tickers[i])
                json_obj = yahoo_financials.get_historical_price_data(beg_date,end_date,"daily")
                ohlv = json_obj[cp_tickers[i]]['prices']
                # now pick formatted_date, open, high, low, close, adjclose
                temp_df = pd.DataFrame(ohlv)[cols]
                temp_df.set_index("formatted_date",inplace=True)
                temp_df.sort_index(inplace = True, ascending=False) 
                ##temp_df = temp_df.sort_index(axis=1 ,ascending=False)
                file_name = conf_rawdata_path + cp_tickers[i] + '.csv'
                print("writing " + file_name)
                temp_df.to_csv(file_name)
                ##print(temp_df.head())
                temp_df.rename(columns={'open' : 'Open', 'high' : 'High', 'low' : 'Low',
                                   'close' : 'Close', 'adjclose' : 'Adj Close',
                                   'volume' : 'Volume'}, inplace=True)
                temp_df.index.names = ['Date']
                reversed_df = temp_df.iloc[::-1]
                file_name = conf_backtest_data_path + cp_tickers[i] + '.csv'
                reversed_df.to_csv(file_name)

                """
                #here is howto add all adjclose + ticker into one table
                #json_obj = yahoo_financials.get_historical_stock_data(beg_date,end_date,"daily")
                ohlv = json_obj[cp_tickers[i]]['prices']
                temp = pd.DataFrame(ohlv)[["formatted_date","adjclose"]]
                temp.set_index("formatted_date",inplace=True)
                temp2 = temp[~temp.index.duplicated(keep='first')]
                close_prices[cp_tickers[i]] = temp2["adjclose"]
                """
                drop.append(cp_tickers[i])       
            except:
                print(cp_tickers[i]," :failed to fetch data...retrying")
                continue
        attempt+=1
    #end of while

def download_etf_prices(years):
    tlist = get_etf_symbols()
    get_price_yahoo(tlist, years)

def download_sector_prices(sector_name, years):
    tlist = get_sector_symbols(sector_name)
    get_price_yahoo(tlist, years)

def download_all_prices(years):
    tlist = get_all_symbols()
    get_price_yahoo(tlist, years)

if __name__ == "__main__":
    download_etf_prices(5)
    ##download_sector_prices("Information Technology",10)
    ##download_sector_prices("Consumer Staples",10)
    ##download_all_prices(10)

