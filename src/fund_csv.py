# ============================================================================
# Getting financial data by url
# Author - Jun Chen
# =============================================================================


#import requests
#from bs4 import BeautifulSoup
import pandas as pd
import datetime

from config import *
from sp500_symbols import *
#import FundamentalAnalysis as fa
#import requests
from urllib.request import urlopen
# import urlopen
import json
#!/usr/bin/env python

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)
    


def financial_ratios(ticker):
    """
    Description
    ----
    Gives information about the financial ratios of a company overtime
    which includes i.a. investment, liquidity, profitability and debt ratios.

    Input
    ----
    ticker (string)
        The company ticker (for example: "LYFT")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    urlstr = ("https://financialmodelingprep.com/api/v3/financial-ratios/" + ticker)
    dj = get_jsonparsed_data(urlstr)
    data_json = dj['ratios']

    data_formatted = {}
    for data in data_json:
        date = data['date'][:4]
        del data['date']
        ratio_data = {}

        for key in data.keys():
            ratio_data.update(data[key])

        data_formatted[date] = ratio_data

    df = pd.DataFrame(data_formatted).transpose()
    df.sort_index(inplace = True, ascending=True)
    df.index.names = ['date']

    return df

def get_fundmental_pe(sector_name, beg_date, end_date):
    tlist = get_sector_symbols(sector_name)
    #tlist = ["AAPL", "AMZN"]
    # Show the available companies
    #companies = fa.available_companies()
    #print(type(companies))

    for ticker in tlist:  
        print('download: '+ticker)
        
        df = financial_ratios(ticker)
        filename = conf_backtest_data_path + ticker+'_ratio.csv'
        # write csv without index column
        df.to_csv(filename, encoding='utf8')

        """
        # Collect general company information
        profile = fa.profile(ticker)
        print(type(profile))
                
        # Collect recent company quotes
        quotes = fa.quote(ticker)
        
        # Collect market cap and enterprise value
        entreprise_value = fa.enterprise(ticker)
        
        # Show recommendations of Analysts
        ratings = fa.rating(ticker)
        
        # Obtain DCFs over time
        dcf_annual = fa.discounted_cash_flow(ticker, period="annual")
        dcf_quarterly = fa.discounted_cash_flow(ticker, period="quarter")
        
        # Collect the Balance Sheet statements
        balance_sheet_annually = fa.balance_sheet_statement(ticker, period="annual")
        balance_sheet_quarterly = fa.balance_sheet_statement(ticker, period="quarter")
        
        # Collect the Income Statements
        income_statement_annually = fa.income_statement(ticker, period="annual")
        income_statement_quarterly = fa.income_statement(ticker, period="quarter")
        
        # Collect the Cash Flow Statements
        cash_flow_statement_annually = fa.cash_flow_statement(ticker, period="annual")
        cash_flow_statement_quarterly = fa.cash_flow_statement(ticker, period="quarter")
        
        # Show Key Metrics
        key_metrics_annually = fa.key_metrics(ticker, period="annual")
        key_metrics_quarterly = fa.key_metrics(ticker, period="quarter")
        
        # Show a large set of in-depth ratios
        financial_ratios = fa.financial_ratios(ticker)
        
        # Show the growth of the company
        growth_annually = fa.financial_statement_growth(ticker, period="annual")
        growth_quarterly = fa.financial_statement_growth(ticker, period="quarter")
        
        # Download general stock data
        stock_data = fa.stock_data(ticker, period="ytd", interval="1d")
        
        # Download detailed stock data
        stock_data_detailed = fa.stock_data_detailed(ticker, begin="2000-01-01", end="2020-01-01")
        """
        
    

def get_fundmental_csv(sector_name, beg_date, end_date):
    tlist = get_sector_symbols(sector_name)
    sum_df = pd.DataFrame()
    for ticker in tlist:  
        print('download: '+ticker)
        #https://stockrow.com/ACN
        #https://github.com/secdatabase/SEC-XBRL-Financial-Statement-Dataset
        #https://github.com/JerBouma/FundamentalAnalysis
        #https://github.com/twopirllc/pandas-ta
        pe_url = 'https://www.quandl.com/api/v3/datatables/SHARADAR/SF1.csv?ticker='+ticker
        pe_url = pe_url + '&qopts.columns=ticker,dimension,datekey,pe,pb,eps,roe&api_key=Jszm-d1BscjHYRVaMmUX'
        #page = requests.get(url)
        #page_content = page.content
        #soup = BeautifulSoup(page_content,'html.parser')
        #req = requests.get(pe_url)  
        #url_content = req.content
        #csv_file = open(ticker+'_pe.csv', 'wb')
        #csv_file.write(url_content)
        #csv_file.close()
        df = pd.read_csv(pe_url)
        sum_df = sum_df.append(df, ignore_index = True)
    
    filename = conf_backtest_data_path + 'finance.csv'
    # write csv without index column
    sum_df.to_csv(filename, index=False, encoding='utf8')
    
if __name__ == "__main__":
    beg_date = datetime.datetime(2010, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.datetime(2012, 1, 1).strftime('%Y-%m-%d')
    get_fundmental_pe("Information Technology",beg_date, end_date)
    #get_fundmental_csv("Information Technology",beg_date, end_date)
    ##download_sector_prices("Consumer Staples",10)
    ##download_all_prices(10)

