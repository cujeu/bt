#!/usr/bin/python
# -*- coding: utf-8 -*-


#from math import ceil
#from lxml import etree
#from urllib.request import urlopen
#import lxml.html
#import pickle
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

def get_etf_symbols():
    etf_list =[ 'FNGU','FNGO','CWEB','TQQQ','ARKW',
                'ARKG','ARKK','TECL','QLD' ,'ROM',
                'ONLN','QCLN','IBUY','FDNI','PBW',
                'ESPO','ARKQ','ARKF','UBOT','OGIG',
                'ACES','TAN' ,'EMQQ','CNRG','SMOG']
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
    create_sp500_csv()
    print(get_sector_symbols("Consumer Discretionary"))

