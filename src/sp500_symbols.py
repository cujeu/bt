#!/usr/bin/python
# -*- coding: utf-8 -*-


#from math import ceil
#from lxml import etree
#from urllib.request import urlopen
#import lxml.html
#import pickle
import os
import datetime
import bs4 as bs
import requests
import pandas as pd

from config import *

def obtain_parse_wiki_snp500():
  """Download and parse the Wikipedia list of S&P500 
  constituents using requests and libxml.
  Returns a list of tuples for to add to MySQL."""

  # Stores the current time, for the created_at record
  now = datetime.datetime.utcnow()

  #page = lxml.html.parse(urlopen('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'))
  resp = requests.get(conf_sp500_url)
  soup = bs.BeautifulSoup(resp.text, 'lxml')
  #  table = soup.find('table', {'class': 'wikitable sortable'})
  symbollist = soup.select('table')[0].select('tr')[1:]
  # Obtain the symbol information for each row in the S&P500 constituent table
  symbols = pd.DataFrame(columns=['ticker', 'instrument', 'name', 'sector', 'industry','currency', 'created_date'])
  for i, symbol in enumerate(symbollist):
    tds = symbol.select('td')
    # Create a tuple (for the DB format) and append to the grand list
    td_list = [tds[0].select('a')[0].text, 'stock', tds[1].select('a')[0].text,
                   tds[3].text,tds[4].text,'USD', now]

    symbols.loc[i] = td_list
    """
    `id` int NOT NULL AUTO_INCREMENT,
    `exchange_id` int NULL,
    `ticker` varchar(32) NOT NULL,
    `instrument` varchar(64) NOT NULL,
    `name` varchar(255) NULL,
    `sector` varchar(255) NULL,
    `currency` varchar(32) NULL,
    `created_date` datetime NOT NULL,
    `last_updated_date` datetime NOT NULL,
    symbols.append(        
        (
            tds[0].select('a')[0].text, #ticker
            'stock', 
            tds[1].select('a')[0].text,  #name
            tds[3].text,  #'sector'], 
            tds[4].text,
            'USD', now 
        )
    )
    """
  return symbols

def insert_snp500_symbols(df_symbols):
    file_name = conf_data_path + conf_sp500_name  #"/home/jun/proj/qalgo/data/sp500.csv"
    df_symbols.to_csv(file_name, encoding='utf-8', index=False)

def create_sp500_csv():
    df_symbols = obtain_parse_wiki_snp500()
    insert_snp500_symbols(df_symbols)
    print("retrieve all symbols")

def read_csv_df():
    file_name = conf_data_path + conf_sp500_name  #"/home/jun/proj/qalgo/data/sp500.csv"
    df_data = pd.read_csv(file_name)
    return df_data

#ark_df is formated df in scnner, now add the changes as new column
def add_ark_diff(ark_df, date_str):
    if date_str == None:
        csv_dir = os.path.join(conf_data_path ,'csv')
        date_list = os.listdir(csv_dir)
        date_list.sort()
        date_str = date_list[-1]
    diff_col = 'diff'
    ark_df[diff_col] = 0
    diff_df = get_ark_diff(date_str)
    ark_list = ark_df['sym'].tolist()

    for index, row in diff_df.iterrows():

        tk = row['ticker']
        diff = row[diff_col]
        if tk in ark_list:
            ark_df.loc[(ark_df.sym == tk),diff_col]=diff
            
            #ark_df.at[tk, diff_col] = diff

    return ark_df

def get_ark_diff(date_picked):
    """
    merge all changes into one df on specified date
    ,date,ticker,diff,diff2marketcap
    0,2020-12-15,TSLA,-281.0,-0.0001768281186816051
    0,2020-12-15,MTLS,35224.0,0.04614838931784272
    ...
    0,2020-12-24,NVDA,1170.0,0.00897875291393339
    0,2020-12-24,AAPL,5976.0,0.03754061679896769
    """
    out_dir = os.path.join(conf_data_path ,'out')
    DATETIME_FORMAT = '%y%m%d'

    # for date_dir in os.listdir(csv_dir):
    out_df = pd.DataFrame()
    csv_date_str = datetime.datetime.strptime(date_picked, DATETIME_FORMAT).strftime('%Y-%m-%d')
    for out_file in os.listdir(out_dir):
        out_file_path = os.path.join(out_dir, out_file)
        extension = os.path.splitext(out_file_path)[1]
        if extension != ".csv":
            continue
        print(out_file_path)
        #read into df
        if len(out_df) > 0:
            tmp_df = pd.read_csv(out_file_path)
            out_df = pd.concat([out_df, tmp_df])
        else:
            out_df = pd.read_csv(out_file_path)

        out_df.drop(out_df[out_df.date != csv_date_str].index, inplace=True)

    out_df.drop(columns=['diff2marketcap'], inplace=True)
    tmp_df = out_df.groupby(['ticker'])['diff'].sum().reset_index()
    #arklist = tmp_df['ticker'].tolist()
    return tmp_df

def get_all_ark_symbol(date_picked):
    csv_dir = os.path.join(conf_data_path ,'csv')
    csv_picked_dir = os.path.join(csv_dir, date_picked)
    if not os.path.exists(csv_picked_dir):
        #find the latest
        csv_dir_list = os.listdir(csv_dir)
        csv_dir_list.sort()
        csv_picked_dir = os.path.join(csv_dir, csv_dir_list[-1])

    arklist = []
    total_list = []
    # for csv_file is holding under dir csv_picked_dir
    for csv_file in os.listdir(csv_picked_dir):
        csv_file_path = os.path.join(csv_picked_dir, csv_file)
        #print(csv_file_path)
        #read into df
        csv_df = pd.read_csv(csv_file_path)
        csv_df.drop(csv_df.tail(3).index, inplace=True)
        csv_df['ticker'] = csv_df['ticker'].astype(str)
        csv_df['ticker'] = csv_df['ticker'].apply(lambda x: x.upper())
        #csv_df.set_index('ticker', inplace=True)
        csv_list = csv_df['ticker'].tolist()
        #arklist.extend(csv_list)
        arklist = list(set(arklist + csv_list))

    #remove some invalid tickers, i.e. ticker from HK...
    for tk in arklist:

        find_digi = False
        #remove space in ticker
        if tk.find(' ') != -1:
            find_digi = True
            #print(tk)
            #total_list.remove(tk)
        else:
            #remove all digit ticker
            for c in tk:
                if c.isdigit():
                    find_digi = True
                    break
        if not find_digi:
            #print(tk)
            total_list.append(tk)

    total_list.sort()
    return total_list

def get_ark_symbols():
    ark_list =[ 'ARKK','ARKW','ARKG','ARKF','ARKQ',
                #ARKK top 2%
                'TSLA','CRSP','ROKU','NVTA','SQ','TDOC','SPOT','PRLB','Z','PSTG','TWLO','IOVA','NTLA','MTLS','VCYT','TSM',
                #ARKW top 2%
                'GBTC','FB','PINS','WORK','SNAP','TCEHY','PD','PYPL','SE','NFLX',
                #ARKG top 2%
                'ARCT','PACB','TWST','EXAS','CDNA','CLLS','PSNL','FATE','MCRB',
                #ARKF top 2%
                'MELI','ICE','BABA','TREE','AAPL','AMZN','DOCU',#'LSPD',
                #ARKQ top 2%
                'GOOG','TRMB','DE','JD','NXPI','FLIR','TWOU','KTOS','IRDM','CAT','WKHS','SSYS','XLNX','AVAV','SPCE','KMTUY']
    return ark_list

def get_etf_symbols():
    etf_list =[ 'FNGU','FNGO','CWEB','TQQQ','ARKW',
                'ARKG','ARKK','TECL','QLD' ,'ROM',
                'ONLN','QCLN','IBUY','FDNI','PBW',
                'ESPO','ARKQ','ARKF','UBOT','OGIG',
                'ACES','TAN' ,'EMQQ','CNRG','SMOG',
                'XLE', 'LIT', 'SKYY']
    etf_list.sort()
    return etf_list

def get_all_symbols():
    df_data = read_csv_df()
    #column name:ticker,instrument,name,sector,currency,created_date
    return df_data['ticker'].values.tolist()

def get_sector_symbols(sector_name):
    df_data = read_csv_df()
    """
    ["Industrials","Health Care","Information Technology",
    "Communication Services","Consumer Discretionary","Utilities",
    "Financials","Materials","Real Estate","Consumer Staples",
    "Energy"]
    """
    return df_data[df_data['sector']==sector_name]['ticker'].tolist()

if __name__ == "__main__":
    print(get_chg_ark_symbol("201224"))
    ##create_sp500_csv()
    ##print(get_sector_symbols("Consumer Discretionary"))

