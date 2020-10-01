from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime
import math
#import os
#import sys  
#import numpy as np
#import random
#import pickle 

import backtrader as bt
import pandas as pd
from config import *
from sp500_symbols import *
from data_maker import *


class FixedCommisionScheme(bt.CommInfoBase):
    '''
    This is a simple fixed commission scheme
    '''
    params = (
        ('commission', 5),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        )

    def _getcommission(self, size, price, pseudoexec):
        """ if commision depends on size
        if abs(size * price) < 10000:
            return 10
        else:
            return 20
        """
        return self.p.commission

# write strategy, pick top3 pe strong stock every hold_days=20 days
class pe_factor_strategy(bt.Strategy):
    author = "jun chen"
    params = (("look_back_days",15),
              ("hold_days",60),
              ("portfolio_size",5))

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    #key_df_list is list of dataFrame of key fundamental data
    def __init__(self, pool_list, key_df_list, dataId_to_ticker_d, ticker_to_dataId_d):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.bar_num=0
        #self.index_50_date_stock_dict=self.get_index_50_date_stock()
        self.dataId_to_ticker_dict = dataId_to_ticker_d
        self.ticker_to_dataId_dict = ticker_to_dataId_d
        self.position_list = []
        self.pool_list = pool_list
        self.key_df_list = key_df_list

    def prenext(self):
        #self.current_date=self.datas[0].datetime.date(0)
        #self.log('prenext :' + str(self.broker.getvalue()), self.current_date)
        #pass 
        self.next()

    def close_position(self):
        # for data in self.datas:
        for stock in self.position_list:
            sec_data = self.getdatabyname(stock)
            
            position_size = self.broker.getposition(data=sec_data).size
            if position_size > 0:
                self.log('try sell:' + sec_data._name)
                self.sell(sec_data, size = position_size)
                #self.close(data)
        self.position_list = []

    def find_swap_position(self):
        # get factor
        result_list=[]
        for stock in self.pool_list:
            kd_idx = self.ticker_to_dataId_dict[stock]
            kdf = self.key_df_list[kd_idx]

            # use pe_raio pe_rate = df_date_data(kdf, stock, 'peRatio', self.current_date.strftime("%Y-%m-%d"))
            pe_rate = df_date_data(kdf, stock, 'roe', self.current_date.strftime("%Y-%m-%d"))

            if not math.isnan(pe_rate):
                result_list.append([stock,pe_rate])

        # find if open or merge new position
        # to_sell_list=[]
        to_buy_list = []
        # self.log("index_300_list:{}".format(index_300_list))
        # for buy position, sort from low growth rate to high growth rate
        sorted_result_list = sorted(result_list,key=lambda x:x[1])

        # to_sell_list=[i[0] for i in sorted_result_list[:self.p.portfolio_size]]
        to_buy_list = [i[0] for i in sorted_result_list[-self.p.portfolio_size:]]
        return to_buy_list

    def open_new_position(self, to_buy_list):
        # get total value (cash + stock)
        # trade may be executed on next day, left some cash space 
        total_value = self.broker.getvalue() * 0.95
        every_stock_value = total_value/self.p.portfolio_size
        for data in self.datas:
            """
            if data._name in to_sell_list:
                close_price=data.close[0]
                lots=int(every_stock_value/close_price)
                self.sell(data,size=lots)
            """    
            if data._name in to_buy_list:
                close_price=data.close[0]
                if close_price > 0:
                    lots=int(every_stock_value/(close_price+0.01))
                    self.buy(data, size=lots, exectype=bt.Order.Market)
                             #valid=bt.Order.DAY)
                    self.position_list.append(data._name)
                    self.log('buy: ' + data._name + str(lots))
        
    def next(self):
        # from cash 100，use 10000 for each swap
        self.bar_num += 1
        self.current_date = self.datas[0].datetime.date(0)
        # time to swap postion
        if self.bar_num % self.p.hold_days != 0:
            return


        self.log('start change position:' + str(self.broker.getvalue()), self.current_date)
            
        # close all position
        self.close_position()

        # make buy list for new posiotn
        to_buy_list = self.find_swap_position()

        # open new position
        self.open_new_position(to_buy_list)

        self.log('end change position', self.current_date)

        
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
    # entry point
    begin_time=time.time()
    cerebro = bt.Cerebro()
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
    
    start_date = datetime.datetime(2010, 1,1)
    #start_date = datetime.datetime(2015, 1,1)
    end_date = datetime.datetime(2019, 11, 29)
    # ticker_list[0] must cover start_date to end_date, as a reference
    ticker_list = get_sector_symbols("Information Technology")
    ticker_list.remove('HPE')
    del ticker_list[-1] #remove the last one, which is SPY
    ticker_list = ["AAPL", "ADBE", "AMZN", "MSFT", "CSCO", "INTC"]
    #print(ticker_list)
    #key_list = ['marketCap','peRatio','pbRatio', 'debtToEquity','roe']
    key_list = ['symbol', 'peRatio','pbRatio','roe']
    key_df_list = []
    dataId_to_ticker_dic = {}
    ticker_to_dataId_dic = {}
    idx = 0
    for tk in ticker_list:
        kdf = key_fundamental_df(tk, key_list,start_date)
        key_df_list.append(kdf)

        trading_data_df = hist_price_df(tk, start_date)
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
    
    #Set commissions
    comminfo = FixedCommisionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    #cerebro.broker.setcommission(commission=0.0001,stocklike=True)
    cerebro.broker.setcash(100000.0)
    cerebro.addstrategy(pe_factor_strategy,
                        ticker_list, key_df_list, dataId_to_ticker_dic, ticker_to_dataId_dic)
                        #end_date.strftime("%Y-%m-%d"))
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    # cerebro.plot()
    end_time=time.time()
    print("total running time:{}".format(end_time-begin_time))
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Multiple Values and Brackets'
        )
    )

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
