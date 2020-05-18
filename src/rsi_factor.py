
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
import pandas as pd
from config import *
from sp500_symbols import *


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

# write strategy, if RSI over 30, then buy 10000 each time
class rsi_factor_strategy(bt.Strategy):
    author = "jun chen"
    params = (("look_back_days",5),
              ("hold_days",15),
              ("buy_cash",10000),
              ("portfolio_size",3),
              ("rsi_upper",70),
              ("rsi_lower",30),
              ('rsi_period', 15))

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

        self.rsi_list = []
        for i in range(len(self.pool_list)):
            rsi = bt.indicators.RSI(self.datas[i], period = self.p.rsi_period)
            self.rsi_list.append(rsi)  #rsi type backtrader.indicators.rsi.RSI
            #rsi = bt.indicators.RSI(period=self.p.rsi_per,upperband=self.p.rsi_upper, lowerband=self.p.rsi_lower)

    def prenext(self):
        #current_date=self.datas[0].datetime.date(0)
        #self.log('prenext :' + str(self.broker.getvalue()), current_date)
        #pass 
        self.next()

    def close_position(self):
        # first , close out if RSI over 70
        close_list = []
        for stock in self.position_list:
            sec_data = self.getdatabyname(stock)
            position_size = self.broker.getposition(data=sec_data).size
            rsi = self.rsi_list[self.ticker_to_dataId_dict[stock]]

            if position_size > 0 and len(rsi) >= self.p.hold_days:
                for idx in range(len(rsi) - self.p.hold_days):
                    if rsi.lines.rsi[0-idx] > self.p.rsi_upper:
                        self.log('try sell:' + sec_data._name)
                        self.sell(sec_data,size=position_size)
                        close_list.append(stock)
                        #self.close(data)
                        break

        for stock in close_list:
            if stock in self.position_list:
                self.position_list.remove(stock)

        return

    def find_swap_position(self):
        # get factor, add stock under rsi30 into result_list
        result_list=[]
        for stock in self.pool_list:
            data = self.getdatabyname(stock)

            #remove un-aligned data
            if (data.datetime.date(0) != self.current_date):
                continue
            if data._name in self.position_list:
                continue

            if (len(data) >= self.p.hold_days):
                rsi = self.rsi_list[self.ticker_to_dataId_dict[stock]]

                #try find new target
                if rsi.lines.rsi[0] > self.p.rsi_lower:
                    for idx in range(self.p.look_back_days):
                        if rsi.lines.rsi[0-idx] < self.p.rsi_lower:
                            result_list.append(stock)
                            break;
        return result_list

    def open_new_position(self, buy_list):
        # get current account value
        # trade may be executed on next day, left some cash space 
        # total_value = self.broker.getvalue() * 0.95
        cur_cash = self.broker.cash
        every_stock_value = self.p.buy_cash * 0.95
        for stock in buy_list:
            data = self.getdatabyname(stock)
            close_price = data.close[0]
            if close_price > 0 and cur_cash > every_stock_value:
                lots=int(every_stock_value/(close_price+0.01))
                self.buy(data, size=lots, exectype=bt.Order.Market)
                         #valid=bt.Order.DAY)
                #self.position_list.append(data._name)
                cur_cash -= every_stock_value
                self.log('buy: ' + data._name + str(lots))
        return

    def next(self):
        # start from 100K cash，buy 10000 when swapping
        self.bar_num+=1
        # current bar time
        self.current_date = self.datas[0].datetime.date(0)

        # exchange
        if self.bar_num % self.p.look_back_days != 0:
            return
        
        self.log('start change position:' + str(self.broker.getvalue()), self.current_date)
        self.close_position()

        buy_list = self.find_swap_position()

        # buy new stock
        self.open_new_position(buy_list)

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
                self.position_list.append(order.data._name)
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

    # entry point
    begin_time=time.time()
    cerebro = bt.Cerebro()
    
    start_date = datetime.datetime(2010, 1,1)
    #start_date = datetime.datetime(2015, 1,1)
    end_date = datetime.datetime(2019, 11, 29)
    # ticker_list[0] must cover start_date to end_date, as a reference
    ticker_list = get_sector_symbols("Information Technology")
    ticker_list.remove('HPE')
    del ticker_list[-1] #remove the last one, which is SPY
    #ticker_list = ["AAPL", "FB", "AMZN", "MSFT", "CSCO"]
    print(ticker_list)
    
    
    dataId_to_ticker_dic = {}
    ticker_to_dataId_dic = {}
    idx = 0
    for tk in ticker_list:
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
    
    #Set commissions
    comminfo = FixedCommisionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    #cerebro.broker.setcommission(commission=0.0001,stocklike=True)
    cerebro.broker.setcash(100000.0)
    cerebro.addstrategy(rsi_factor_strategy,
                        ticker_list, dataId_to_ticker_dic, ticker_to_dataId_dic)
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
        description=('Multiple Values and Brackets')
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
