import os
import pandas as pd
import datetime
from tiingo import TiingoClient

from config import *
from sp500_symbols import *

def get_today_price_tiingo(all_tickers, tclient):
    #all_tickers = ["GOOG","ARKK"]
    # extracting stock data (historical close price) for the stocks identified

    #close_prices = pd.DataFrame()
    today = datetime.datetime.today().date()
    shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
    end_date = (today - shift + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    beg_date = end_date  ##.strftime('%Y-%m-%d') # (datetime.date.today()-datetime.timedelta(1))
    #beg_date = (datetime.date.today()-datetime.timedelta(365*years)).strftime('%Y-%m-%d')

    cp_tickers = all_tickers
    attempt = 0
    drop = []
    # names in cols are features of get_historical_price_data
    reorder_cols = ["open", "high", "low", "close","adjclose","volume"]
    # used_cols = ["formatted_date","open", "high", "low", "close","adjclose","volume"]
    unused_cols = ['adjHigh','adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor']
 
    #cols = ["Date","Open", "High", "Low", "Close","Adj Close","Volume"]
    # other order: Date,Open,High,Low,Close,Adj Close,Volume
    # anther order Date,Open,High,Low,Close,Volume,Adj Close
    while len(cp_tickers) != 0 and attempt < 2:
        print("-----------------")
        print("attempt number ",attempt)
        print("-----------------")
        cp_tickers = [j for j in cp_tickers if j not in drop]
        for i in range(len(cp_tickers)):
            try:
                """
                dataframe columns
                Index(['close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh',
                       'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'],
                 ->formatted_date,open,high,low,close,adjclose,volume
                """

                temp_df = client.get_dataframe( cp_tickers[i],startDate=beg_date, endDate=end_date, frequency='daily')
                if len(temp_df) < 1:
                    continue
                
                #only left ohlc format
                temp_df.drop(columns=unused_cols, inplace=True)

                #reorder column sequence
                temp_df.rename(columns={'adjClose' : 'adjclose'}, inplace=True)
                temp_df = temp_df[reorder_cols]
                #temp_df.Index.set_names(['formatted_date'], inplace=True)
                temp_df.index.names = ['formatted_date']

                # now set formatted_date, open, high, low, close, adjclose, volume 
                #temp_df.set_index("formatted_date",inplace=True)
                #temp_df.sort_index(inplace = True, ascending=False) 
                ##temp_df = temp_df.sort_index(axis=1 ,ascending=False)
                file_name = conf_rawdata_path + cp_tickers[i] + '.csv'
                csv_df = pd.read_csv(file_name)
                csv_df.set_index("formatted_date",inplace=True)
                #lastrow = [end_date.strftime('%Y-%m-%d')]
                #lasrrow.append(temp_df.values[-1].tolist())
                #csv_df.loc[len(csv_df)] = lastrow
                frames = [temp_df, csv_df]
                temp_df = pd.concat(frames)
                
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
                #json_obj = iex_financials.get_historical_stock_data(beg_date,end_date,"daily")
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

def download_etf_prices(years, tclient):
    tlist = get_etf_symbols()
    get_today_price_tiingo(tlist, tclient)

def download_all_ark_prices(years, tclient):
    DATETIME_FORMAT = '%y%m%d'
    date_str = datetime.datetime.strftime(datetime.datetime.now(), DATETIME_FORMAT)
    tlist = get_all_ark_symbol(date_str)
    #print(tlist)
    get_today_price_tiingo(tlist, tclient)

def download_ark_prices(years, tclient):
    tlist = get_ark_symbols()
    get_today_price_tiingo(tlist, tclient)

def download_sector_prices(sector_name, years, tclient):
    tlist = get_sector_symbols(sector_name)
    get_today_price_tiingo(tlist, tclient)

def download_all_prices(years, tclient):
    tlist = get_all_symbols()
    get_today_price_tiingo(tlist, tclient)

if __name__ == "__main__":
    ##download_sector_prices("Information Technology",10)
    ##download_sector_prices("Consumer Staples",10)
    #download_ark_prices(10)
    tiingo_token = os.getenv('TIINGO_TOKEN')
    config = {}
    # To reuse the same HTTP Session across API calls (and have better performance), include a session key.
    config['session'] = True
    # pass TIINGO_TOKEN in via a configuration dictionary.
    config['api_key'] = tiingo_token
    # Initialize
    client = TiingoClient(config)

    ## tiingo limits
    ## 500 unique symbols per month, 500 requests/hour, 20,000 requests/day, 5 GB/mth (source
    #download_all_prices(10,client)
    download_etf_prices(10,client)
    #download_all_ark_prices(10,client)


