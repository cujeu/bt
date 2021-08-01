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

g_entry_list = []
g_alert_list = []
# write strategy
class atr_scrn_strategy(bt.Strategy):
    author = "jun chen"
    params = (("look_back_days",120),
              ("hold_days",15),
              ("portfolio_size",5))

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self, pool_list, dataId_to_ticker_d, ticker_to_dataId_d,start_date,end_date):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.bar_num=0
        #self.index_50_date_stock_dict=self.get_index_50_date_stock()
        self.pool_list = pool_list
        self.dataId_to_ticker_dict = dataId_to_ticker_d
        self.ticker_to_dataId_dict = ticker_to_dataId_d
        self.start_date = start_date
        self.end_date = end_date
        self.position_list = [] #postion order ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"]
        self.csCount = 0
        self.barDiv = 0
        self.periodTD9 = 12
        self.pastTD9 = [0] * 20
        self.downTD9 = 0

        self.hasPosition = False
        self.entryPrice = 100
        self.highPrice = 100
        self.newLowDiv = 100
        self.newLowBar = 0
        self.atrTrend = False
        self.curStrategy = 0 #0:idle 10:divergence 11: div + 20ma turn up div+20ma is what happen after divergence
                             #20:atr

        global g_entry_list
        g_entry_list = []

        # add indicators
        for d in self.datas:
            ## d.baseline = btind.ExponentialMovingAverage(d, period=21)
            d.ema20 = btind.ExponentialMovingAverage(d, period=20)
            d.ema60 = btind.ExponentialMovingAverage(d, period=60)
            d.macd = btind.MACD(d)
            d.priceDiv = PriceDiv(d)
            d.data_obv = OnBalanceVolume(d)
            d.data_obv20 = btind.SMA(d.data_obv, period=20)
            d.data_atr = AtrTrend(d)
            #d.data_bb = btind.BollingerBands(d, period=20, devfactor=2.0)
            #d.data_kc = KeltnerChannel(d, periodEMA = 20, periodATR = 20, devfactor=1.5 )
            #d.data_rg = RGChannel(d)

            self.log('stategy ATR init :' + d._name)

    def stop(self):
        if self.hasPosition:
            stock = self.pool_list[0]
            data = self.getdatabyname(stock)
            self.hasPosition = False
            result_list = self.position_list[-1]
            del self.position_list[-1]
                # ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"]
            result_list[3] = str(round(self.highPrice,2))
            result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
            result_list[5] = str(round(data.close[0],2))
            result_list[6] = str(self.current_date)
            result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
            result_list[8] = str(self.curStrategy)
            self.position_list.append(result_list)

        global g_entry_list
        g_entry_list = self.position_list
        """
        if len(self.position_list) > 0:
            self.log('last div low:' + self.position_list[-1])
        """

    def prenext(self):
        #self.current_date=self.datas[0].datetime.date(0)
        #self.log('prenext :' + str(self.broker.getvalue()), self.current_date)
        #pass 
        self.next()

    def test_exit_position(self):
        result_list=[]
        if not self.hasPosition:
            return result_list
        if len(self.pool_list) > 0:
            stock = self.pool_list[0]
            data = self.getdatabyname(stock)

            #no processing of un-aligned data
            if (data.datetime.date(0) != self.current_date):
                return result_list;   #continue

            # get the price delta during period of look_back_days
            if len(data) < self.p.look_back_days :
                return result_list;  ##continue
                #self.log('test :' + stock + str(self.divUnder)+ ' ema20=' +str(data.ema20[0]) + ' ema20=' +str(data.ema20[-2]))
                
            #if (data.priceDiv.cs[0] < data.priceDiv.sm[0] and 
            #    data.priceDiv.cs[-1] > data.priceDiv.sm[-1]):
            
            # in normal process 10 --> 11 --> 20 --> 0
            if (self.curStrategy >= 10) and (self.curStrategy <= 12):
                lastStr = self.curStrategy
                # ema20 turn up and close over ema20
                if (data.close[0] > data.ema20[0]) and \
                   (data.ema20[0] > data.ema20[-1] and data.ema20[-2] > data.ema20[-1]):
                    self.curStrategy = 11

                if (data.ema20[0] > data.ema20[-1] and data.ema20[-2] > data.ema20[-1]):
                    for x in range(self.periodTD9):
                        #self.log('test :' + str(self.current_date) + str(self.pastTD9[(self.bar_num - x) % self.periodTD9]))
                        if self.pastTD9[(self.bar_num - x) % self.periodTD9] >= 9:
                            self.curStrategy = 12   ## 12 is TD9 then ema up
                            break
                    #self.log('test :' + str(self.current_date) + str(self.curStrategy)+ ' ema20=' +str(data.ema20[0]) + ' ema20=' +str(data.ema20[-1]))

                if (self.curStrategy != lastStr):
                    self.barDiv = 0
                    self.entryPrice = data.close[0]
                    self.highPrice = data.close[0]
                    result_list = self.position_list[-1]
                    del self.position_list[-1]
                    result_list[1] = str(self.current_date)
                    result_list[2] = str(round(data.close[0], 2))
                    result_list[3] = str(round(self.highPrice, 2))
                    result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
                    result_list[5] = str(round(data.close[0], 2))
                    result_list[6] = str(self.current_date)
                    result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
                    result_list[8] = str(self.curStrategy)
                    self.position_list.append(result_list)
                    result_list = []
                    if (self.current_date + datetime.timedelta(days=3)) >= self.end_date:
                        global g_alert_list
                        g_alert_list.append([stock,str(self.curStrategy),str(self.current_date)])

            if ((self.curStrategy >= 10) and (self.curStrategy <= 12)) and \
               (data.priceDiv.cs[0] < 0 and data.priceDiv.cs[-1] > 0 and \
                data.priceDiv.cs[0] < data.priceDiv.sm[0]) :

                #transfer the strategy
                if self.atrTrend:
                    self.curStrategy = 20 + self.curStrategy - 10
                else:
                    self.hasPosition = False
                    result_list = self.position_list[-1]
                    del self.position_list[-1]
                    # ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"]
                    result_list[3] = str(round(self.highPrice, 2))
                    result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
                    result_list[5] = str(round(data.close[0], 2))
                    result_list[6] = str(self.current_date)
                    result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
                    result_list[8] = str(self.curStrategy)
                    self.position_list.append(result_list)
                    self.curStrategy = 0
                    self.barDiv = 0

            elif self.curStrategy >= 20 and (not self.atrTrend):
                self.hasPosition = False
                result_list = self.position_list[-1]
                del self.position_list[-1]
                # ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"]
                result_list[3] = str(round(self.highPrice, 2))
                result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
                result_list[5] = str(round(data.close[0], 2))
                result_list[6] = str(self.current_date)
                result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
                result_list[8] = str(self.curStrategy)
                self.curStrategy = 0
                self.position_list.append(result_list)

        return result_list


    def test_entry_position(self):
        # get factor
        result_list=[]
        if self.hasPosition:
            return result_list
        if len(self.pool_list) > 0:
            stock = self.pool_list[0]
            data = self.getdatabyname(stock)

            #no processing of un-aligned data
            if (data.datetime.date(0) != self.current_date):
                return result_list;   #continue

            # get the price delta during period of look_back_days
            if len(data) < self.p.look_back_days :
                return result_list;  ##continue
                #self.log('test :' + stock + str(self.divUnder)+ ' ema20=' +str(data.ema20[0]) + ' ema20=' +str(data.ema20[-2]))
                
            cond_atr = False
            cond_div = False
            cond_TD9U = False
            if self.curStrategy == 0:
                cond_atr = self.atrTrend
                cond_div = ((data.priceDiv.cs[0] < -10 or
                            (data.priceDiv.cs[0] < -5 and 
                             data.priceDiv.cs[0] > self.newLowDiv and
                             (self.bar_num - self.newLowBar) > 1)) and
                            data.priceDiv.cs[0] > data.priceDiv.cs[-1])
                if cond_div:
                    #search past to find TD9
                    cond_div = False
                    for x in range(5): #self.periodTD9):
                        if self.pastTD9[(self.bar_num - x) % self.periodTD9] >= 9:
                            cond_div = True
                            break

                if (data.priceDiv.cs[0] < 0) and (data.ema20[0] > data.ema20[-1] and data.ema20[-2] > data.ema20[-1]):
                    for x in range(self.periodTD9):
                        if self.pastTD9[(self.bar_num - x) % self.periodTD9] >= 9:
                            self.curStrategy = 30   ## 30 is TD9 then ema up
                            cond_TD9U = True
                            cond_atr = False
                            cond_div = False
                            break

                #if (cond_atr and (data.priceDiv.cs[0] > data.priceDiv.sm[0] and
                #                 data.priceDiv.cs[0] < data.priceDiv.ml[0] and
                #                 data.priceDiv.sm[0] > data.priceDiv.sm[-1])) or 

                if cond_atr or cond_div or cond_TD9U:
                    if cond_div:
                        self.curStrategy = 10
                        self.barDiv = self.bar_num
                    elif cond_TD9U:
                        self.curStrategy = 30
                    else:
                        self.curStrategy = 20

                    self.hasPosition = True
                    self.entryPrice = data.close[0]
                    self.highPrice = data.close[0]
                    result_list.append(stock)
                    result_list.append(str(self.current_date))
                    result_list.append(str(round(data.close[0],2)))
                    result_list.append(str(round(data.close[0],2))) #high
                    result_list.append("0")                         #perc
                    result_list.append(str(round(data.close[0],2))) #exit
                    result_list.append(str(self.current_date))      #edate
                    result_list.append("0")                         #eperc
                    result_list.append(str(self.curStrategy))       #strategy

                            #self.log('new div low :' + stock + \
                            #         ' date:' + str(self.current_date) + \
                            #         ' div:' + str(self.newLowDiv))

        # find new low div
        return result_list

    def next(self):
        self.bar_num += 1
        self.current_date = self.datas[0].datetime.date(0)
        if self.bar_num < self.p.look_back_days :
            return

        data = self.getdatabyname(self.pool_list[0])        
        if data.close[0] < data.close[-4]:
            self.downTD9 += 1
            #self.downTD9 = self.downTD9 % self.periodTD9
        else:
            self.downTD9 = 0
        self.pastTD9 [self.bar_num % self.periodTD9] = self.downTD9

        if data.priceDiv.cs[0] >= 0:
            if (self.csCount > 1):
                self.csCount -= 2
            elif (self.csCount > 0):
                self.csCount -= 1
        else:
            self.csCount += 1
        if data.priceDiv.cs[0] < self.newLowDiv :
            self.newLowDiv = data.priceDiv.cs[0]
            self.newLowBar = self.bar_num

        mindn = min(self.data.data_atr.dn[-5], min(self.data.data_atr.dn[-6], self.data.data_atr.dn[-2]))
        maxup = max(self.data.data_atr.up[-5], max(self.data.data_atr.up[-6], self.data.data_atr.up[-2]))
        if (self.atrTrend and self.data.close[0] < maxup): #data.data_atr.up[-1]) :
            self.atrTrend = False
        elif (self.atrTrend==False and self.data.close[0] > mindn): #data.data_atr.dn[-1]):
            self.atrTrend = True


        if self.hasPosition:
            if data.close[0] > self.highPrice:
                self.highPrice = data.close[0]
            to_watch_list = self.test_exit_position()
            if len(to_watch_list) > 0 :
                self.hasPosition = False
        else:
            to_watch_list = self.test_entry_position()

            if len(to_watch_list) > 0 :
                self.hasPosition = True
                self.position_list.append(to_watch_list)
                #gap = datetime.timedelta(3)   #max(1,(today.weekday() + 6) % 7 - 3))
                if (self.current_date + datetime.timedelta(days=3)) >= self.end_date:
                    global g_alert_list
                    g_alert_list.append([to_watch_list[0],str(self.curStrategy),str(self.current_date)])
                    self.log('recent position:' + to_watch_list[0] + ' '+str(self.curStrategy)+ ' '+str(self.current_date))


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
    mk_df = pd.DataFrame(columns = ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"])
    for ticker in ticker_list:
    
        # entry point
        begin_time=time.time()
        cerebro = bt.Cerebro()
        dataId_to_ticker_dic = {}
        ticker_to_dataId_dic = {}
        idx = 0
        cerebro_ticker_list = []
        cerebro_ticker_list.append(ticker)
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
                break
            trading_data_df.drop(['Adj Close'], axis=1, inplace=True)
            trading_data_df.index.names = ['date']
            trading_data_df.rename(columns={'Open' : 'open', 'High' : 'high', 'Low' : 'low',
                                            'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
            ## set data range by date
            indexDates = trading_data_df[trading_data_df.index < start_tm].index
            # Delete these row indexes from dataFrame
            trading_data_df.drop(indexDates , inplace=True)
            trading_data_df['openinterest'] = 0
            #trading_data_df.set_index('Date', inplace=True)

            #__init__ is called at time of feeding data
            data_feed = bt.feeds.PandasData(dataname=trading_data_df,
                                            fromdate=start_date,
                                            todate=end_date)  # dtformat=('%Y-%m-%d'))
            cerebro.adddata(data_feed, name=tk)
            dataId_to_ticker_dic.update({idx:tk})
            ticker_to_dataId_dic.update({tk:idx})
            idx += 1
        
        if not len_enough:
            continue
        
        cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%
        #cerebro.broker.set_slippage_fixed(1, slip_open=True)
        
        #cerebro.broker.setcommission(commission=0.0001,stocklike=True)
        cerebro.broker.setcash(init_cash)

        cerebro.addstrategy(atr_scrn_strategy,
                            cerebro_ticker_list, dataId_to_ticker_dic, ticker_to_dataId_dic,
                            start_date, end_date)
        #cerebro.addindicator(SchaffTrend)
        #print('Starting Scanning')
        cerebro.run()
        # cerebro.plot()
        end_time=time.time()
        #print("total running time:{}".format(end_time-begin_time))
        #print('Ending [%s]', ', '.join(map(str,mk_view_list)))
        print('+++')
        #print('')
        
        #add list to dataframe
        for e in g_entry_list:
            # e is list of ["sym", "date", "close", "high", "perc", "exit","edate","eperc","strategy"]
            mk_df.loc[len(mk_df)] = e
        #final_value += cerebro.broker.getcash() + cerebro.broker.getvalue - init_cast
        #final_value = final_value + cerebro.broker.getcash() + cerebro.broker.getvalue() - init_cash
        profit_value = profit_value + cerebro.broker.getvalue() - init_cash

    #end of for loop of ticker list
    print(mk_df)
    print ("       !!!!!!!!!!!    ")
    print(g_alert_list)
    print ("++++++++++  ")
    return mk_df

def runstrat(sector_name, today):
    #today = datetime.datetime.today().date()
    shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
    end_date = today - shift + datetime.timedelta(days=1)
    #start_date = datetime.datetime(2015, 1,1)
    start_date = end_date - datetime.timedelta(days=4*365)
    # ticker_list[0] must cover start_date to end_date, as a reference

    ## 1. scan ETF
    if sector_name == 'all':
        ticker_list = get_etf_symbols()
        #ticker_list = ['TECL','FNGU','FNGO','CWEB','TQQQ','ARKW','ARKG','ARKK','QLD' ,'ROM']
        #ticker_list = ['ARKK','ARKF','TECL']
        g_alert_list.append(["ETF==>",str(end_date)])
        mk_df = start_scan(ticker_list, start_date, end_date)
        filename = conf_data_path + 'scan_etf.csv'
        mk_df.to_csv(filename, encoding='utf8')

        ## 2.scan ARK
        #ticker_list = get_ark_symbols()
        DATETIME_FORMAT = '%y%m%d'
        csv_dir = os.path.join(conf_data_path ,'csv')
        csv_list = os.listdir(csv_dir)
        csv_list.sort()
        if len(csv_list) > 0:
            date_str = csv_list[-1]
        else:
            date_str = datetime.datetime.strftime(datetime.datetime.now(), DATETIME_FORMAT)

        ticker_list = get_all_ark_symbol(date_str)
        g_alert_list.append(["ARK==>",str(end_date)])
        mk_df = start_scan(ticker_list, start_date, end_date)
        filename = conf_data_path + 'scan_ark.csv'
        mk_df.to_csv(filename, encoding='utf8')
        #with open(filename, 'a') as f:
        #    mk_df.to_csv(f, header=False)
        #    f.close

    ## 3. scan russell
    if sector_name == 'all':
        ticker_list = get_strong_russell_symbols()
    elif sector_name == 'grow':
        ticker_list = get_growth_russell_symbols()
    else:
        ticker_list = get_russell_symbols_by_sector(sector_name)
    g_alert_list.append(["RUS==>",str(end_date)])
    mk_df = start_scan(ticker_list, start_date, end_date)
    ## get ticker list and then add new sector column
    tlist = mk_df["sym"].tolist()
    slist = get_sec_by_symlist(tlist)
    mk_df['sector'] = slist
    filename = conf_data_path + 'scan_russell.csv'
    mk_df.to_csv(filename, encoding='utf8')

    print("scan done")

    """
    ## 3.scan all sp500
    ticker_list = get_all_symbols()
    mk_df = start_scan(ticker_list, start_date, end_date)
    filename = conf_data_path + 'scan_sp500.csv'
    mk_df.to_csv(filename, encoding='utf8')
    #with open(filename, 'a') as f:
    #    mk_df.to_csv(f, header=False)
    #    f.close
    """

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(description='ATR Startegy')

    # Defaults for dates
    # example: python atr_srn -s all grow sector
    parser.add_argument('--sector', '-s', required=False, default='all',
                        help='russell sectors')
    parser.add_argument('--industry', '-i', required=False, default='all',
                        help='russell industry')
    parser.add_argument('--date', '-d', required=False, default='none',
                        help='target date')


    return parser.parse_args(pargs)
 
if __name__ == '__main__':
    args = parse_args()
    if args.date != 'none':
        today = datetime.datetime.strptime(args.date, '%Y-%m-%d')
        today = today.date()
    else:
        today = datetime.datetime.today().date()
    runstrat(args.sector,today)
    """
    for a, d in ((getattr(args, x), x) for x in ['sector', 'industry']):
        if d == 'sector':
            runstrat(args.sector)
            #print(get_russell_symbols_by_sector(args.sector))
        elif d == 'industry':
            print(a)
    """

