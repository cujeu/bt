# price divergence factor for ETF poll
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
import os
import argparse
#import sys  
#import numpy as np
#import random
#import pickle 

import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
from config import *
from sp500_symbols import *
from inst_ind import *
        
mk_view_list = []
g_newlow_list = []
g_ema_count = 0

# write strategy
class div_ema_scan_strategy(bt.Strategy):
    author = "jun chen"
    params = (("look_back_days",75),
              ("hold_days",15),
              ("portfolio_size",5))

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self, pool_list, dataId_to_ticker_d, ticker_to_dataId_d,start_date,end_date, ref_price):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.bar_num=0
        #self.index_50_date_stock_dict=self.get_index_50_date_stock()
        self.dataId_to_ticker_dict = dataId_to_ticker_d
        self.ticker_to_dataId_dict = ticker_to_dataId_d
        self.position_list = []
        self.pool_list = pool_list
        self.start_date = start_date
        self.end_date = end_date
        self.ref_price = ref_price
        self.newLowDiv = 100
        self.newLowIndex = 0
        self.csCount = 0

        #mk_view_list order [ticker, cs, sm, ml]
        global mk_view_list
        mk_view_list = []
        mk_view_list.append(pool_list[0])
        for x in range(len(conf_mk_view_col)-1):
            mk_view_list.append(0)

        for d in self.datas:
            ## d.baseline = btind.ExponentialMovingAverage(d, period=21)
            d.ema20 = btind.ExponentialMovingAverage(d, period=20)
            d.ema60 = btind.ExponentialMovingAverage(d, period=60)
            d.priceDiv = PriceDiv(d)
            self.log('stategy Divergence init :' + d._name)

    def stop(self):
        stock = self.pool_list[0]
        data = self.getdatabyname(stock)
        global g_ema_count
        global mk_view_list
        """
        emaDown = 0
        for x in range(15):
            if data.ema20[-x] < data.ema20[-x-1]:
                emaDown += 10
        for x in range(3):
            if data.ema20[-x] > data.ema20[-x-1]:
                emaDown += 1
        """
        for x in range(3):
            if (data.ema20[-x] > data.ema20[-x-1]) and (data.ema20[-x-2] > data.ema20[-x-1]):
                g_ema_count += 1
        mk_view_list[conf_chg5_idx] = round(100 * (data.close[0] - data.close[-5]) / (data.close[-5] + 0.01),2)
        mk_view_list[conf_chg10_idx] = round(100 * (data.close[0] - data.close[-10]) / (data.close[-10] + 0.01),2)
        mk_view_list[conf_cs0Cnt_idx] = self.csCount
        mk_view_list[conf_cs_idx] = round(data.priceDiv.cs[0],2)
        mk_view_list[conf_csChg_idx] = round(data.priceDiv.cs[0] - data.priceDiv.cs[-1], 2)
        mk_view_list[conf_sm_idx] = round(data.priceDiv.sm[0],2)
        mk_view_list[conf_ml_idx] = round(data.priceDiv.ml[0],2)
        mk_view_list[conf_ema20_idx] = round(data.ema20[0],2)
        mk_view_list[conf_ema20Chg_idx] = round(data.ema20[0] - data.ema20[-1],2)
        mk_view_list[conf_ema60_idx] = round(data.ema60[0],2)
        mk_view_list[conf_date_idx] = str(self.end_date)
        mk_view_list[conf_close_idx] = round(data.close[0],2)
        mk_view_list[conf_ref_idx] = round(self.ref_price,2)

        if len(self.position_list) > 0:
            self.log('last div low:' + self.position_list[-1])

    def prenext(self):
        #self.current_date=self.datas[0].datetime.date(0)
        #self.log('prenext :' + str(self.broker.getvalue()), self.current_date)
        #pass 
        self.next()


    def find_new_div_low(self):
        # get factor
        result_list=[]
        for stock in self.pool_list:
            data = self.getdatabyname(stock)

            #no processing of un-aligned data
            if (data.datetime.date(0) != self.current_date):
                continue

            # get the price delta during period of look_back_days
            if len(data) >= self.p.look_back_days :
                #self.log('test :' + stock + str(self.divUnder)+ ' ema20=' +str(data.ema20[0]) + ' ema20=' +str(data.ema20[-2]))
                
                # entry point: cs back and ema keep up
                #if data.priceDiv.cs[0] > data.priceDiv.sm[0] and \
                #   data.priceDiv.cs[-1] <= data.priceDiv.sm[-1] and \
                if (data.priceDiv.cs[0] > 0):
                    self.newLowDiv = data.priceDiv.cs[0]
                else:
                    if (data.priceDiv.cs[0] > data.priceDiv.cs[-1] and \
                        data.priceDiv.cs[0] < self.newLowDiv):
                        ##data.ema60[0] > data.ema60[-1]):
                            result_list.append(stock)
                            self.newLowDiv = data.priceDiv.cs[0]
                            self.newLowIndex = self.bar_num
                            self.position_list.append(str(self.current_date))
                            if data.priceDiv.cs[-1] < -5:
                                if (self.current_date + datetime.timedelta(days=3)) >= self.end_date:
                                    global g_newlow_list
                                    g_newlow_list.append([stock, str(self.current_date)])
                                    self.log('new div low :' + stock + \
                                             ' date:' + str(self.current_date) + \
                                             ' div:' + str(self.newLowDiv))

        # find new low div
        return result_list

    def next(self):
        # from cash 100，use 10000 for each swap
        self.bar_num += 1
        self.current_date = self.datas[0].datetime.date(0)
        if self.bar_num < self.p.look_back_days :
            return

        data = self.getdatabyname(self.pool_list[0])        
        if data.priceDiv.cs[0] >= 0:
            if (self.csCount > 1):
                self.csCount -= 2
            elif (self.csCount > 0):
                self.csCount -= 1
        else:
            self.csCount += 1

        to_watch_list = self.find_new_div_low()

        #if len(to_watch_list) > 0 :
        #    if (self.current_date + datetime.timedelta(days=3)) >= self.end_date:
        #        global g_newlow_list
        #        g_newlow_list.append([to_watch_list[0], str(self.current_date)])
        #    self.log('new low div position:' + str(self.current_date))

    def notify_order(self, order):
        #Created, Submitted, Accepted, Partial, Completed, \
        #Canceled, Expired, Margin, Rejected = range(9)
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            # dt=order.created.dt
            # self.log('ORDER ACCEPTED/SUBMITTED '+ dt.strftime("%Y-%m-%d"))
            #self.log('ORDER ACCEPTED/SUBMITTED '+ str(self.bar_num))
            self.order = order
            return

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('BUY EXPIRED' + str(order.status))

            # Check if an order has been completed
            # Attention: broker could reject order if not enougth cash
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY %s EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.data._name,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            else:  # Sell
                self.log('SELL %s EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.data._name,
                          order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                
    def notify_trade(self, trade):
        if trade.isclosed:
            
            self.log('TRADE %s PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.data._name, trade.pnl,  trade.pnlcomm))

def start_scan(ticker_list, start_date, end_date):
    profit_value = 1.0;
    init_cash = 100000.0
    start_tm = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day,)
    #mk_df = pd.DataFrame([["a",1, 2, 3]], columns = ["sym", "cs", "sm", "ml"])
    mk_df = pd.DataFrame(columns = conf_mk_view_col)
    for ticker in ticker_list:
    
        # entry point
        # begin_time=time.time()
        cerebro = bt.Cerebro()
        dataId_to_ticker_dic = {}
        ticker_to_dataId_dic = {}
        idx = 0
        cerebro_ticker_list = []
        cerebro_ticker_list.append(ticker)
        ref_price = 0
        len_enough = True
        for tk in cerebro_ticker_list:
            filename = conf_backtest_data_path + tk + '.csv'
            if not os.path.exists(filename):
                len_enough = False
                break
            trading_data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            #check row count
            if trading_data_df.shape[0] < 150:
                len_enough = False
                break;
            trading_data_df.drop(['Adj Close'], axis=1, inplace=True)
            trading_data_df.index.names = ['date']
            trading_data_df.rename(columns={'Open' : 'open', 'High' : 'high', 'Low' : 'low',
                                            'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
            ref_price = trading_data_df.iloc[-1]['close']
            ## set data range by date
            indexDates = trading_data_df[trading_data_df.index < start_tm].index
            # Delete these row indexes from dataFrame
            trading_data_df.drop(indexDates , inplace=True)
            trading_data_df['openinterest'] = 0
            #trading_data_df.set_index('Date', inplace=True)
            data_feed = bt.feeds.PandasData(dataname=trading_data_df,
                                            fromdate=start_date,
                                            todate=end_date)  # dtformat=('%Y-%m-%d'))
            cerebro.adddata(data_feed, name=tk)
            dataId_to_ticker_dic.update({idx:tk})
            ticker_to_dataId_dic.update({tk:idx})
            idx += 1
        
        if not len_enough:
            ## add empty
            global mk_view_list
            mk_view_list = []
            mk_view_list.append(tk)
            for x in range(len(conf_mk_view_col)-1):
                mk_view_list.append(0)
            mk_df.loc[len(mk_df)] = mk_view_list
            print('new stock ' + tk)
            continue
        
        global g_ema_count
        g_ema_count += 1000
        cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%
        #cerebro.broker.set_slippage_fixed(1, slip_open=True)
        
        #cerebro.broker.setcommission(commission=0.0001,stocklike=True)
        cerebro.broker.setcash(init_cash)

        cerebro.addstrategy(div_ema_scan_strategy,
                            cerebro_ticker_list, dataId_to_ticker_dic, ticker_to_dataId_dic,
                            start_date, end_date, ref_price)
                            #end_date.strftime("%Y-%m-%d"))
        #cerebro.addindicator(SchaffTrend)
        #print('Starting Scanning')
        results = cerebro.run()
        # cerebro.plot()
        end_time=time.time()
        #print("total running time:{}".format(end_time-begin_time))
        #print('Ending [%s]', ', '.join(map(str,mk_view_list)))
        print('+++')
        
        #add list to dataframe
        mk_df.loc[len(mk_df)] = mk_view_list
        #final_value += cerebro.broker.getcash() + cerebro.broker.getvalue - init_cast
        #final_value = final_value + cerebro.broker.getcash() + cerebro.broker.getvalue() - init_cash
        profit_value = profit_value + cerebro.broker.getvalue() - init_cash

    #end of for loop
    print(mk_df)
    print("===================")
    return mk_df

    
def runstrat(sector_name, today):
    global g_ema_count
    #today = datetime.datetime.today().date()
    shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
    end_date = today - shift + datetime.timedelta(days=1)
    #start_date = datetime.datetime(2015, 1,1)
    start_date = end_date - datetime.timedelta(days=5*365)-datetime.timedelta(days=1)

    #default sector_name = all
    if sector_name == 'all':
        """
        #1. watch sp500

        ticker_list = get_all_symbols()
        mk_df = start_scan(ticker_list, start_date, end_date)
        filename = conf_data_path + 'div_sp500.csv'
        mk_df.to_csv(filename, encoding='utf8')
        # ticker_list[0] must cover start_date to end_date, as a reference
        """

        #2. watch ETFs
        ticker_list = get_etf_symbols()
        #ticker_list = ['TECL','FNGU','FNGO','CWEB','TQQQ','ARKW','ARKG','ARKK','QLD' ,'ROM']
        #ticker_list = ['TECL', 'FNGU','ARKK']
        #print(ticker_list)
        g_ema_count = 0
        mk_df = start_scan(ticker_list, start_date, end_date)
        g_newlow_list.append(["<==ETF",str(g_ema_count) + "|"+str(end_date)])
        print(g_newlow_list)
        filename = conf_data_path + 'div_etf.csv'
        mk_df.to_csv(filename, encoding='utf8')

        """  ARK use excel format now
        #3. watch ARK
        DATETIME_FORMAT = '%y%m%d'
        csv_dir = os.path.join(conf_data_path ,'csv')
        csv_list = os.listdir(csv_dir)
        csv_list.sort()
        if len(csv_list) > 0:
            date_str = csv_list[-1]
        else:
            date_str = datetime.datetime.strftime(datetime.datetime.now(), DATETIME_FORMAT)

        ticker_list = get_all_ark_symbol(date_str)
        g_ema_count = 0
        mk_df = start_scan(ticker_list, start_date, end_date)
        g_newlow_list.append(["<==ARK",str(g_ema_count) + "|"+str(end_date)])
        print(g_newlow_list)
        # diff column is latest day position change
        mk_df = add_ark_diff(mk_df, date_str)

        filename = conf_data_path + 'div_ark.csv'
        mk_df.to_csv(filename, encoding='utf8')
        """

    #4. watch russell
    #hard code for growth ,
    if sector_name == 'all':
        sector_name = 'grow'
    if sector_name == 'all':
        ticker_list = get_strong_russell_symbols()
    elif sector_name == 'grow':
        #watch growth russell
        ticker_list = get_growth_russell_symbols()
    elif sector_name == 'spec':
        #watch and find something in russell
        ticker_list = get_spec_russell_symbols()
    else:
        ticker_list = get_russell_symbols_by_sector(sector_name)

    g_ema_count = 0
    mk_df = start_scan(ticker_list, start_date, end_date)
    g_newlow_list.append(["<==RUS",str(g_ema_count) + "|"+str(end_date)])
    print(g_newlow_list)

    filename = conf_data_path + 'div_russell.csv'
    ## get ticker list and then add new sector column
    tlist = mk_df["sym"].tolist()
    slist = get_sec_by_symlist(tlist)
    mk_df['sector'] = slist
    mk_df.to_csv(filename, encoding='utf8')
    # ticker_list[0] must cover start_date to end_date, as a reference

    # Open a file with access mode 'a' to append results
    log_file = open('daily.txt', 'a')
    now_date = datetime.datetime.now().strftime("%m/%d/%Y")
    log_file.write('\n=====  start of %s ===== total|ema-up \n' %now_date)
    for listitem in g_newlow_list:
        log_file.write('%s\n' % listitem)
    log_file.close()

    print("scan done")

 
def parse_args(pargs=None):
    parser = argparse.ArgumentParser(description='divergence watch')

    # Defaults for dates
    parser.add_argument('--sector', '-s', required=False, default='all',
                        help='russell sectors')
    parser.add_argument('--industry', '-i', required=False, default='all',
                        help='russell industry')
    parser.add_argument('--date', '-d', required=False, default='none',
                        help='target date,2021-3-22')


    return parser.parse_args(pargs)

def mkdate(datestr):
    return time.strptime(datestr, '%Y-%m-%d')


if __name__ == '__main__':
    args = parse_args()
    if args.date != 'none':
        today = datetime.datetime.strptime(args.date, '%Y-%m-%d')
        today = today.date()
    else:
        today = datetime.datetime.today().date()
    runstrat(args.sector, today)

    """
    print(args.sector, args.industry)
    for a, d in ((getattr(args, x), x) for x in ['sector', 'industry']):
        if d == 'sector':
            runstrat(args.sector)
        elif d == 'industry':
            print(a)
    """
