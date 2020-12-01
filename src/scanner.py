# price divergence factor for ETF poll
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
#import os
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

    def __init__(self, pool_list, dataId_to_ticker_d, ticker_to_dataId_d):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.bar_num=0
        #self.index_50_date_stock_dict=self.get_index_50_date_stock()
        self.dataId_to_ticker_dict = dataId_to_ticker_d
        self.ticker_to_dataId_dict = ticker_to_dataId_d
        self.position_list = []
        self.pool_list = pool_list
        self.newLowDiv = 100
        self.newLowIndex = 0

        for d in self.datas:
            ## d.baseline = btind.ExponentialMovingAverage(d, period=21)
            d.ema20 = btind.ExponentialMovingAverage(d, period=20)
            d.ema60 = btind.ExponentialMovingAverage(d, period=60)
            d.priceDiv = PriceDiv(d)
            self.log('stategy data init :' + d._name)

    def stop(self):
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
                            #self.log('new div low :' + stock + \
                            #         ' date:' + str(self.current_date) + \
                            #         ' div:' + str(self.newLowDiv))

        # find new low div
        return result_list

    def next(self):
        # from cash 100，use 10000 for each swap
        self.bar_num += 1
        self.current_date = self.datas[0].datetime.date(0)
        if self.bar_num < self.p.look_back_days :
            return
        #for stock in self.pool_list:
        #    data = self.getdatabyname(stock)
        
        to_watch_list = self.find_new_div_low()

        #if len(to_watch_list) > 0 :
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

def runstrat(args=None):
    """
    args = parse_args(args)

    # Data feed kwargs
    kwargs = dict()

    # Parse from/to-date
    dtfmt, tmfmt = '%Y-%m-%d', 'T%H:%M:%S'
    for a, d in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        if a:
            strpfmt = dtfmt + tmfmt * ('T' in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)

    # Data feed
    data0 = bt.feeds.YahooFinanceCSVData(dataname=args.data0, **kwargs)
    cerebro.adddata(data0, name='d0')

    data1 = bt.feeds.YahooFinanceCSVData(dataname=args.data1, **kwargs)
    data1.plotinfo.plotmaster = data0
    cerebro.adddata(data1, name='d1')

    data2 = bt.feeds.YahooFinanceCSVData(dataname=args.data2, **kwargs)
    data2.plotinfo.plotmaster = data0
    cerebro.adddata(data2, name='d2')

    # Broker
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
    cerebro.broker.setcommission(commission=0.001)

    # Sizer
    # cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))
    cerebro.addsizer(TestSizer, **eval('dict(' + args.sizer + ')'))

    # Strategy
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))

    # Execute
    cerebro.run(**eval('dict(' + args.cerebro + ')'))

    if args.plot:  # Plot if requested to
        cerebro.plot(**eval('dict(' + args.plot + ')'))
    """

    """
    # the oringinal code, usign CSV and data fiel
    joinquant_day_kwargs = dict(
                fromdate = datetime.datetime(2005, 1,1),
                todate = datetime.datetime(2019, 11, 29),
                timeframe = bt.TimeFrame.Days,
                compression = 1,
                dtformat=('%Y-%m-%d'),
                datetime=0,
                high=4,
                low=3,
                open=1,
                close=2,
                volume=5,
                openinterest=6,
                factor=7,
                pb_ratio=10,
                pe_ratio=11)
    
    data_path="/home/yun/data/stocks/"
    with open("/home/yun/data/all_index_50_stock_list.pkl",'rb') as f:
        file_list=pickle.load(f)
    file_list=[i+'.csv' for i in file_list]
    file_list=file_list
    print(file_list)
    count=0
    for file in file_list:
        print(count,file)
        count+=1
        feed = GenericCSV_PB_PE(dataname = data_path+file, **joinquant_day_kwargs)
        cerebro.adddata(feed, name = file[:-4])
    """
    
    start_date = datetime.datetime(2015, 1,1)
    #start_date = datetime.datetime(2015, 1,1)
    end_date = datetime.datetime(2020, 11, 1)
    # ticker_list[0] must cover start_date to end_date, as a reference
    ticker_list = get_etf_symbols()
    #ticker_list = ['TECL','FNGU','FNGO','CWEB','TQQQ','ARKW','ARKG','ARKK','QLD' ,'ROM']
    ticker_list = ['TECL', 'FNGU','ARKK']
    print(ticker_list)
    profit_value = 1.0;
    init_cash = 100000.0
    
    for ticker in ticker_list:
    
        # entry point
        begin_time=time.time()
        cerebro = bt.Cerebro()
        dataId_to_ticker_dic = {}
        ticker_to_dataId_dic = {}
        idx = 0
        cerebro_ticker_list = []
        cerebro_ticker_list.append(ticker)
        for tk in cerebro_ticker_list:
            filename = conf_backtest_data_path + tk + '.csv'
            trading_data_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            trading_data_df.drop(['Adj Close'], axis=1, inplace=True)
            trading_data_df.index.names = ['date']
            trading_data_df.rename(columns={'Open' : 'open', 'High' : 'high', 'Low' : 'low',
                                            'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
            ## set data range by date
            indexDates = trading_data_df[trading_data_df.index < start_date].index
            # Delete these row indexes from dataFrame
            trading_data_df.drop(indexDates , inplace=True)
            trading_data_df['openinterest'] = 0
            #trading_data_df.set_index('Date', inplace=True)
            data_feed = bt.feeds.PandasData(dataname=trading_data_df,
                                            #fromdate = datetime.datetime(2010, 1, 4),
                                            #todate = datetime.datetime(2019, 12, 31))
        
                                            fromdate=start_date,
                                            todate=end_date)  # dtformat=('%Y-%m-%d'))
            cerebro.adddata(data_feed, name=tk)
            dataId_to_ticker_dic.update({idx:tk})
            ticker_to_dataId_dic.update({tk:idx})
            idx += 1
        
        
        cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%
        #cerebro.broker.set_slippage_fixed(1, slip_open=True)
        
        #cerebro.broker.setcommission(commission=0.0001,stocklike=True)
        cerebro.broker.setcash(init_cash)
        cerebro.addstrategy(div_ema_scan_strategy,
                            cerebro_ticker_list, dataId_to_ticker_dic, ticker_to_dataId_dic)
                            #end_date.strftime("%Y-%m-%d"))
        #cerebro.addindicator(SchaffTrend)
        print('Starting Scanning')
        results = cerebro.run()
        # cerebro.plot()
        end_time=time.time()
        print("total running time:{}".format(end_time-begin_time))
        print('Ending ')
        print('+++')
        print('')
        #final_value += cerebro.broker.getcash() + cerebro.broker.getvalue - init_cast
        #final_value = final_value + cerebro.broker.getcash() + cerebro.broker.getvalue() - init_cash
        profit_value = profit_value + cerebro.broker.getvalue() - init_cash

    #end of for loop
    print ("++++++++++  ")

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description=('Multiple Values and Brackets'))

    parser.add_argument('--data0', default='../data/backtest/AAPL.csv',
                        required=False, help='Data0 to read in')

    parser.add_argument('--data1', default='../data/backtest/AMZN.csv',
                        required=False, help='Data1 to read in')

    parser.add_argument('--data2', default='../data/backtest/MSFT.csv',
                        required=False, help='Data1 to read in')

    # Defaults for dates
    parser.add_argument('--fromdate', required=False, default='2011-01-01',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--todate', required=False, default='2017-01-01',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

    parser.add_argument('--cerebro', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--broker', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--sizer', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    runstrat()
