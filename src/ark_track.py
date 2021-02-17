import os
import math
#import datetime
#import requests
import pandas as pd
from config import *

def makedir(dirname):
    #dir = os.path.join(os.getcwd(), dirname)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileExistsError as e:
        pass
    return dirname

os.system("rm -rf " + os.path.join(conf_data_path ,'out'))
os.system("rm -rf " + os.path.join(conf_data_path ,'pickle'))

out_dir = makedir( os.path.join(conf_data_path ,'out'))
pickle_dir = makedir( os.path.join(conf_data_path ,'pickle'))
csv_dir = makedir( os.path.join(conf_data_path ,'csv'))

DATETIME_FORMAT = '%y%m%d'

def rewrite_pickle(from_date_dir):

    date_dir = from_date_dir
    pickle_dir_list = os.listdir(pickle_dir)
    pickle_dir_list.sort()

    #process if need to compare difference pickle files

    #now date_dir_path is new directory
    date_dir_path = os.path.join(csv_dir, date_dir)
    for csv_file in os.listdir(date_dir_path):
        csv_file_path = os.path.join(date_dir_path, csv_file)
        csv_name = csv_file.split('.')[0]
        
        #read into df
        #print(csv_file_path)
        csv_df = pd.read_csv(csv_file_path)

        #drop the unnessary 3 lines in tail
        csv_df.drop(csv_df.tail(3).index, inplace=True)
        csv_df['ticker'] = csv_df['ticker'].astype(str)
        csv_df['ticker'] = csv_df['ticker'].apply(lambda x: x.upper())
        csv_df.set_index('ticker', inplace=True)

        shares_df = csv_df.filter(['shares'])
        shares_df = shares_df.T
        shares_df.index = [pd.to_datetime(date_dir, format=DATETIME_FORMAT )]

        all_shares_path = os.path.join(pickle_dir, csv_name+'-shares.pickle')
        try:
            all_shares_df = pd.read_pickle(all_shares_path)
        except FileNotFoundError as e:
            all_shares_df = pd.DataFrame()

        shares_df = shares_df.drop(['NAN'], axis=1, errors='ignore')

        all_shares_df = pd.concat([all_shares_df, shares_df])
        all_shares_df.sort_index(inplace=True)

        all_shares_df.to_pickle(all_shares_path)

        marketcap_df = csv_df.filter(['market value($)'])
        marketcap_df = marketcap_df.T
        marketcap_df.index = [pd.to_datetime(date_dir, format=DATETIME_FORMAT )]

        all_marketcap_path = os.path.join(pickle_dir, csv_name+'-marketcap.pickle')

        try:
            all_marketcap_df = pd.read_pickle(all_marketcap_path)
        except FileNotFoundError as e:
            all_marketcap_df = pd.DataFrame()

        marketcap_df = marketcap_df.drop(['NAN'], axis=1, errors='ignore')

        all_marketcap_df = pd.concat([all_marketcap_df, marketcap_df])
        all_marketcap_df.sort_index(inplace=True)

        all_marketcap_df.to_pickle(all_marketcap_path)


def output_changes(date_dir):
    etfs = set()
    for e in os.listdir(pickle_dir):
        esp = e.split('-')  #esp = ['ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS', 'marketcap.pickle']
        if len(esp) > 1:
            etfs.add(esp[0])
    for etf in etfs:
        # sbuf = etf.split('.')
        shares_df = pd.read_pickle(os.path.join(pickle_dir, etf+'-shares.pickle'))
        marketcap_df = pd.read_pickle(os.path.join(pickle_dir, etf+'-marketcap.pickle'))
    
        shares_df_diff = shares_df.diff().to_dict()
        marketcap_df_diff = marketcap_df.diff().to_dict()
        for ticker in shares_df_diff:
            lbuf = list(shares_df_diff[ticker].values())
            for i in range(0, len(lbuf)):
                if lbuf[i] != 0 and not math.isnan(lbuf[i]):
                    marketcap_diff = list(marketcap_df_diff[ticker].values())[i]
                    # find share difference
                    etf_path = os.path.join(pickle_dir, etf+'-changes.pickle')
                    try:
                        etf_df = pd.read_pickle(etf_path)
                    except FileNotFoundError as e:
                        etf_df = pd.DataFrame()
    
                    # find changes of market cap 
                    data = {'date': [list(shares_df_diff[ticker].keys())[i]], 
                            'ticker': [ticker],
                            'diff': [lbuf[i]],
                            'diff2marketcap': [lbuf[i]/marketcap_df[ticker].to_list()[i]*100]}
                    tmp_df = pd.DataFrame(data=data)
                    etf_df = pd.concat([etf_df, tmp_df])
                    etf_df.to_pickle(etf_path)
    
    """
    #remove the indicator(i.e.201219) from pickle_dir
    for oldfile in os.listdir(pickle_dir):
        if len(oldfile.split('.')) < 2:
            os.remove(os.path.join(pickle_dir, oldfile))
    """
    
    #write the changes into out directory
    for pickle in os.listdir(pickle_dir):
        if 'changes.pickle' in pickle.split('-'):
            df = pd.read_pickle(os.path.join(pickle_dir, pickle))
            df.to_csv(os.path.join(out_dir, pickle.split('.')[0]+'.csv'))
    
    """
    #add latest dir_name(i.e.201219)  into pickle_dir as indicator
    indicator = os.path.join(pickle_dir, date_dir)
    open(indicator, 'a').close()
    """

csv_list = os.listdir(csv_dir)
csv_list.sort()
for date_dir in csv_list:
    print("process " + date_dir)
    rewrite_pickle(date_dir)

date_dir = csv_list[-1]
#write the changes
output_changes(date_dir)

print("Done")


