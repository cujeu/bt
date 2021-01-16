import os
import math
import datetime
import requests
import pandas as pd
from config import *

DOWNLOAD = True
# DOWNLOAD = False
#REPICKLE = True

def makedir(dirname):
    #dir = os.path.join(os.getcwd(), dirname)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileExistsError as e:
        pass
    return dirname

out_dir = makedir( os.path.join(conf_data_path ,'out'))
csv_dir = makedir( os.path.join(conf_data_path ,'csv'))
pickle_dir = makedir( os.path.join(conf_data_path ,'pickle'))

DATETIME_FORMAT = '%y%m%d'

def get_csv_write_pickle():
    # get the last date of read url
    latest = None
    csv_dir_list = os.listdir(csv_dir)
    csv_dir_list.sort()
    if len(csv_dir_list) > 0:
        latest = csv_dir_list[-1]
        print('csv latest =' + latest)

    urls = arklinks
    #read from web and out into csv directory
    for url in urls:
        date_str = datetime.datetime.strftime(datetime.datetime.now(), DATETIME_FORMAT)
        date_dir = makedir(os.path.join(csv_dir, date_str))
        #url is https://xxx, then 'requests' content
        local_file = os.path.join(date_dir, url.split('/')[-1])
        if not os.path.exists(local_file):
            fark = open(local_file, "wb")
            fark.write(requests.get(url).content)
            fark.close()

    # make a file list from pickle_dir
    # pickle_dir include '201213' as the latest time to run ark_monitor
    # then do 'pickle' compare from that day
    """
    pickle_dir_list = os.listdir(pickle_dir)
    pickle_dir_list.sort()
    for pickle_file in pickle_dir_list:
        if len(pickle_file.split('.')) < 2:
            #latest = i.e '201213'
            #latest = datetime.datetime.strptime(pickle_file, DATETIME_FORMAT)
            latest = pickle_file
            break
    print('latest is ' + latest)
    """

    #process if need to compare difference pickle files
    for date_dir in os.listdir(csv_dir):
        if latest:
            #if datetime.datetime.strptime(date_dir, DATETIME_FORMAT) <= latest:
            if date_dir <= latest:
                continue

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


def out_changes():
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
    
    #remove the indicator(i.e.201219) from pickle_dir
    """
    for file in os.listdir(pickle_dir):
        if len(file.split('.')) < 2:
            os.remove(os.path.join(pickle_dir, file))
    """
    
    #write the changes into out directory
    for pickle in os.listdir(pickle_dir):
        if 'changes.pickle' in pickle.split('-'):
            df = pd.read_pickle(os.path.join(pickle_dir, pickle))
            df.to_csv(os.path.join(out_dir, pickle.split('.')[0]+'.csv'))
    
    """
    #add latest dir_name(i.e.201219)  into pickle_dir as indicator
    csv_date_list = os.listdir(csv_dir)
    csv_date_list.sort()
    indicator = os.path.join(pickle_dir, csv_date_list[-1])
    open(indicator, 'a').close()
    """

# download csv from https://ark-funds.com/wp-content/fundsiteliterature/csv
if DOWNLOAD:
    get_csv_write_pickle()

#write the changes
out_changes()

# for date in date_dirs:
#     date_datetime = datetime.datetime.strptime(date, DATETIME_FORMAT)
#     if latest is None | date_datetime > latest:
#         print(date)
#         date_dir = os.path.join(csv_dir, date)
#         if os.path.isdir(date_dir):
#             for csv_file in os.listdir(date_dir):
#                 csv_path = os.path.join(date_dir, csv_file)
#                 # ft_pickle_path = os.path.join(
#                 #     pickle_dir, csv_file.split('.')[0]+'-'+datetime.datetime.strftime(date_datetime)+'.pickle')
#                 latest_pickle = os.path.join(pickle_dir, csv_file.split('.')[
#                     0]+'-'+datetime.datetime.strftime(latest, DATETIME_FORMAT)+'.pickle')
#                 if os.path.isfile(latest_pickle):
#                     ftdf = pd.read_pickle(latest_pickle)
#                 else:
#                     ftdf = pd.DataFrame()
#                 csvdf = pd.read_csv(csv_path)
#                 csvdf.drop(csvdf.tail(3).index, inplace=True)
#                 csvdf['ticker'] = csvdf['ticker'].astype(str)
#                 csvdf.set_index('ticker', inplace=True)
#                 csvdf = csvdf.filter(['shares'])
#                 csvdf = csvdf.T
#                 csvdf.index = [pd.to_datetime(date)]
#                 for ticker in csvdf.columns.to_list():
#                     if ticker not in ftdf and ticker in csvdf.columns.to_list():
#                         if ticker.lower() not in ['nan']:
#                             ftdf[ticker] = ""
#                         else:
#                             del csvdf[ticker]

#             #         | TSLA | AAPL | ...
#             # 200820  | 1111 |   69 | ...
#             # 210820  |   11 |   12 | ...
#             #   .     |    . |    . | ...
#             #   .     |    . |    . | ...

#             ftdf = pd.concat([ftdf, csvdf])
#             ftdf.to_pickle(os.path.join(
#                 pickle_dir, csv_file.split('.')[0]+'-'+date+'.pickle'))
#             ftdf.to_csv(csv_file)
# done = set()
# for csv in csv_files:
#     csv_name = csv.split('-')[0]
#     if csv_name not in done:
#         done.add(csv_name)
#         csv_path = os.path.join(csv_dir, csv)
#         pickle_path = os.path.join(pickle_dir, csv_name+'.pickle')
#         date = datetime.datetime.strptime(
#             csv.split('-')[-1].split('.')[0], DATETIME_FORMAT)
#         if not os.path.isfile(pickle_path):
#             df = pd.DataFrame()
#             df.to_pickle(pickle_path)
#         else:
#             df = pd.read_pickle(pickle_path)
# df['ticker'] =
# df = pd.read_csv(csv_path)
# df.drop(df.tail(3).index, inplace=True)
# print(df)
# pdfs = [x.split("/")[-1] for x in files]
# symbols = set()
# for pdf in pdfs:
#     pdf_path = os.path.join(tmp_dir, pdf)
#     if dl:
#         raw = parser.from_file(pdf_path)
#         txt = raw['content']
#         open(pdf_path.split('.')[0] + '.txt', 'w').write(txt)
#     else:
#         txt = open(pdf_path.split('.')[0] + '.txt', 'r').read()
#     txt = [x for x in txt.split('\n')[42:-27] if x]
#     from datetime import datetime
#     date = datetime.strptime(txt[0].split(' ')[-1], '%m/%d/%Y')
#     titles = ['Ticker', 'CUSIP', 'Shares', 'Market Value', 'Weight']
#     txt = txt[2:]
#     dataset = [row.split("   ")[1].split(" ")[-5:] for row in txt]
#     df = pd.DataFrame(data=dataset, columns=titles)
#     df['Shares'] = df['Shares'].str.replace(',', '').astype(int)
#     df['Market Value'] = df['Market Value'].str.replace(
#         ',', '').astype(float)
#     df['Weight'] = df['Weight'].astype(float)
#     df['S2MV'] = df['Shares']/df['Market Value']
#     if dl_stock_csv:
#         for sym in list(df['Ticker'].to_dict().values()):
#             if sym not in symbols:
#                 symbols.add(sym)
#                 import time
#                 now = int(time.time())
#                 then = now - 31622400
#                 url = f"https://query1.finance.yahoo.com/v7/finance/download/{sym}?period1={then}&period2={now}&interval=1d&events=history"
#                 r = requests.get(url)

#                 csv_folder = os.path.join(tmp_dir, "stock_csv")
#                 try:
#                     os.mkdir(csv_folder)
#                 except FileExistsError as e:
#                     pass
#                 sym_csv = os.path.join(csv_folder, sym) + "-" + \
#                     str(then) + "-"+str(now)+".csv"
#                 open(sym_csv, "wb").write(r.content)
#                 dfsym = pd.read_csv(sym_csv)
#                 if not dfsym.empty:
#                     # dfsym.to_csv(sym_csv)
#                     dfsym.index = pd.to_datetime(dfsym['Date'])
#                 else:
#                     os.remove(sym_csv)
#                     print(f"empty: {sym}    {url}")

#     csv_path = os.path.join(
#         csv_dir, f"{pdf.split('.')[0]}-{datetime.strftime(date,'%m%d%y')}.csv")
#     open(csv_path, 'a').close()
#     df.to_csv(path_or_buf=csv_path)
