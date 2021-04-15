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

def read_russell_csv_df():
    file_name = conf_data_path + russell1000_name
    df_data = pd.read_csv(file_name)
    return df_data

def write_russell_csv_df(df_data):
    file_name = conf_data_path + russell1000_name
    df_data.to_csv(file_name, index=False)
    return

#ark_df is formated df in scaner, now add the changes as new column
def add_ark_diff(ark_df, date_str):
    date_list = []
    ## if date_str == None:
    csv_dir = os.path.join(conf_data_path ,'csv')
    date_list = os.listdir(csv_dir)
    date_list.sort()
    ark_list = ark_df['sym'].tolist()

    """
    add columns below
    sym     diff1	diff2	diff3	diff4	diff5   mom
    aapl    28199	95682	35595	36697	22841   5/6
    accd    25389	37107	52731	82026	13671   5/6
    """
    for x in range(1,6):
        date_str = date_list[0 - x]
        diff_col = 'diff'+str(x)
        ark_df[diff_col] = 0

        diff_df = get_ark_diff(date_str)

        for index, row in diff_df.iterrows():
            #column ate,ticker,diff,diff2marketcap
            tk = row['ticker']
            diff = row['diff']
            if tk in ark_list:
                ark_df.loc[(ark_df.sym == tk),diff_col] = diff
                #ark_df.at[tk, diff_col] = diff

    pos_rate_col = 'pos'
    ark_df[pos_rate_col] = 0
    for index, row in ark_df.iterrows():
        pos_rate = 0
        tk = row['sym']
        for x in range(1,6):
            diff_col = 'diff'+str(x)
            if row[diff_col] > 0:
                pos_rate += 1
        
        #count momentu buying power
        r = 0
        if pos_rate > 0:
            r = int(100 * pos_rate / 5)
        ark_df.loc[(ark_df.sym == tk),pos_rate_col] = r

    return ark_df

def get_ark_diff(date_picked):
    """
    merge all changes into one df on specified date
    ,date,ticker,diff,diff2marketcap
    0,2020-12-15,TSLA,-281.0,-0.0001768281186816051
    0,2020-12-16,TSLA,3355.0,0.002104745564491819
    0,2020-12-17,TSLA,2013.0,0.0012739111443980909
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
        #print(out_file_path)
        #read into df
        if len(out_df) > 0:
            tmp_df = pd.read_csv(out_file_path)
            tmp_df.drop(tmp_df[tmp_df.date != csv_date_str].index, inplace=True)
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

    #remove some non-US tickers, i.e. ticker from HK...
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
    etf_list =[ 'ARKK','ARKG','ARKQ','ARKF','ARKW',
                'TECL','FNGU','SOXX','QLD' ,'ROM',
                'ONLN','QCLN','IBUY','FDNI','PBW',
                'ESPO','CWEB','TQQQ','UBOT','OGIG',
                'ACES','TAN' ,'EMQQ','CNRG','SMOG',
                'QQQ', 'LIT', 'SKYY','BOTZ','HAIL',
                'XLU', 'XLV', 'XLF', 'XLE', 'XLRE',
                'XLB', 'XLC', 'XLI', 'XLP', 'XTL']
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

def get_info_by_sym(ticker_name):
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    # column names
    #Symbol,Name,Industry,"52W %Chg","Market Cap",Sales(a),"Net Income(a)",Sector,"5Y Rev%",ROE%,Debt/Equity,"Price/Cash Flow"
    tlist = []
    for index, row in df_data.iterrows():
        s = row['Symbol']
        if s == ticker_name:
            tlist.append(row['Sector'])
            tlist.append(row['Industry'])
            tlist.append(row['Market Cap'])
            tlist.append(row['Net Income(a)'])
            tlist.append(row['5Y Rev%'])
            tlist.append(row['ROE%'])
            tlist.append(row['Debt/Equity'])
            tlist.append(row['Price/Cash Flow'])
            break;
    return tlist

def get_sec_by_symlist(ticker_list):
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    slist = []
    for t in ticker_list:
        slist.append("NA")
        for index, row in df_data.iterrows():
            s = row['Symbol']
            if s == t:
                info = row['Sector'] + '/' + row['Industry']
                slist[-1] = info
                break
    return slist

def get_russell_symbols():
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    #tlist = df_data['Symbol'].values.tolist()
    tlist = []
    for index, row in df_data.iterrows():
        s = row['Symbol']
        #remove invalid char [. space] in symbol
        if s.find('.') < 0 and s.find(' ') < 0:
            tlist.append(s)
    return tlist

def get_growth_russell_symbols():
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    #upper bound cap 18B , lower bound 2B(here the unit is 1million)
    tlist = []
    capLarge = 25000.0
    capSmall =  2000.0
    for index, row in df_data.iterrows():
        cap = row['Market Cap']
        if cap < capSmall or cap > capLarge:
            continue
        s = row['Symbol']
        #remove invalid char [. space] in symbol
        if s.find('.') < 0 and s.find(' ') < 0:
            tlist.append(s)
    return tlist

# top 20 companies, 
# 1. sales  > 100M  2. net income >500M
# 3. 5Y Rev% >50% 4. Debt/Equity < 2% 5. sorted by 5Y Rev%
def get_strong_russell_symbols():
    df_data = read_russell_csv_df()
    
    tlist = []
    indexNames = df_data[df_data['Sales(a)'] < 100].index
    df_data.drop(indexNames , inplace=True)
    indexNames = df_data[df_data['Net Income(a)'] < 100].index
    df_data.drop(indexNames , inplace=True)
    #df_data[["5Y Rev%"]] = df_data[["5Y Rev%"]].apply(pd.to_numeric)
    indexNames = df_data[df_data['5Y Rev%'] < 50].index
    df_data.drop(indexNames , inplace=True)
    indexNames = df_data[df_data['Debt/Equity'] > 2].index
    df_data.drop(indexNames , inplace=True)

    #df_data.to_csv('/home/jun/temp/test.csv', encoding='utf-8', index=False)
    tlist = df_data["Symbol"].tolist()
    return tlist

def get_russell_symbols_by_sector(sector_name):
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    #upper bound cap 18B , lower bound 2B
    tlist = []
    for index, row in df_data.iterrows():
        sec = row['Sector']
        if sec == sector_name:
            s = row['Symbol']
            #remove invalid char [. space] in symbol
            if s.find('.') < 0 and s.find(' ') < 0:
                tlist.append(s)

    return tlist

# list stock symbols by industry
def get_russell_symbols_by_ind(ind_name):
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    tlist = []
    for index, row in df_data.iterrows():
        sec = row['Industry']
        if sec == ind_name:
            s = row['Symbol']
            #remove invalid char [. space] in symbol
            if s.find('.') < 0 and s.find(' ') < 0:
                tlist.append(s)

    return tlist

#list all industry names
def get_russell_industry():
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    # pick Industry column from 
    #Symbol,Name,Industry,"52W %Chg","Market Cap",Sales(a),"Net Income(a)",Sector
    slist = []
    for index, row in df_data.iterrows():
        indu = row['Industry']
        if not indu in slist:
            slist.append(indu)
    return slist

#list all sector names
def get_russell_sectors():
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    # pick sector column from 
    #Symbol,Name,Industry,"52W %Chg","Market Cap",Sales(a),"Net Income(a)",Sector
    slist = []
    for index, row in df_data.iterrows():
        sec = row['Sector']
        if not sec in slist:
            slist.append(sec)
    return slist

#list industries of a sector
def get_russell_industry_sector(sector_name):
    df_data = read_russell_csv_df()
    
    #drop the unnessary 2 lines in tail
    #df_data.drop(df_data.tail(2).index, inplace=True)
    
    # pick sector column from 
    #Symbol,Name,Industry,"52W %Chg","Market Cap",Sales(a),"Net Income(a)",Sector
    slist = []
    for index, row in df_data.iterrows():
        sec = row['Sector']
        if sec == sector_name:
            indu = row['Industry']
            if not indu in slist:
                slist.append(indu)
    return slist

# make data frame clean
def div1000000(x):
    try:
        x = float(x)
        return x / 1000000
    except AttributeError:
        return 0

def refine_russell_data():
    df_data = read_russell_csv_df()
    #drop the unnessary 2 lines in tail
    df_data.drop(df_data.tail(2).index, inplace=True)

    # Delete these row indexes contain invalid symbol (BRK.B) from dataFrame
    indexNames = df_data[ df_data['Symbol'] == 'GOOGL' ].index
    df_data.drop(indexNames , inplace=True)
    indexNames = df_data[ df_data['Symbol'].str.contains("\.")].index
    df_data.drop(indexNames , inplace=True)
    # "Symbol	Name	Industry	52W %Chg	Market Cap	Sales(a)	Net Income(a)	Sector	5Y Rev%	ROE%	Debt/Equity	Price/Cash Flow"
    #df[1] = df[1].apply(add_one)
    df_data['52W %Chg'] = df_data['52W %Chg'].apply(lambda x: x.strip('%'))
    df_data['5Y Rev%'] = df_data['5Y Rev%'].apply(lambda x: x.replace(',',''))
    df_data['5Y Rev%'] = df_data['5Y Rev%'].apply(lambda x: x.strip('%'))
    df_data['ROE%'] = df_data['ROE%'].apply(lambda x: x.strip('%'))

    df_data['Market Cap'] = df_data['Market Cap'].apply(lambda x: x/1000000)
    df_data['Sales(a)'] = df_data['Sales(a)'].apply(lambda x: div1000000(x))
    df_data['Net Income(a)'] = df_data['Net Income(a)'].apply(lambda x: div1000000(x))
    write_russell_csv_df(df_data)
    
    return

if __name__ == "__main__":

    print(get_strong_russell_symbols())
    #print(get_russell_sectors())
    #print(get_russell_industry_sector('Medical'))

    #print(get_growth_russell_symbols())
    ##print(get_chg_ark_symbol("201224"))
    ##create_sp500_csv()
    ##print(get_sector_symbols("Consumer Discretionary"))

