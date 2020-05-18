
# 按照标准动量策略做一个多因子回测的模板，开始测试各个单个因子能够带来超额alpha
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

"""
reference from https://zhuanlan.zhihu.com/p/100252474
1,2,3,4,5代表的是波动率依次从低到高进行排序的10支股票的组合，
可以看到，总体上，低波动率的组合在前期是高于高波动率的组合的，
在2016年之后，高波动率组合表现亮眼，远远超过其他四组
"""

"""
# 在交易信息之外，额外增加了PE、PB指标，做其他策略使用
class GenericCSV_PB_PE(bt.feeds.GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('factor','pe_ratio','pb_ratio',)

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('factor', 8),('pe_ratio',9),('pb_ratio',10),)

# 在交易信息之外，额外增加了PE、PB指标，做其他策略使用
import pickle
with open("/home/yun/data/index_300_stock_list.pkl",'rb') as f:
    date_stock_list=pickle.load(f)
"""

# 编写策略
class momentum_factor_strategy(bt.Strategy):
    author = "jun chen"
    params = (("look_back_days",15),
              ("hold_days",15),
              ("portfolio_size",3))

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

    def prenext(self):
        #current_date=self.datas[0].datetime.date(0)
        #self.log('prenext :' + str(self.broker.getvalue()), current_date)
        #pass 
        self.next()
        
    def next(self):
        # 假设有10万资金，每次成份股调整，每个股票使用1万元
        self.bar_num+=1
        # 需要调仓的时候
        if self.bar_num % self.p.hold_days == 0:
            # 得到当天的时间
            current_date=self.datas[0].datetime.date(0)
            self.log('start change position:' + str(self.broker.getvalue()), current_date)
            # 获得当天的交易的股票
            index_50_list = self.pool_list #dataId_to_ticker_dict[str(current_date)]
            # 先全部平仓
            # for data in self.datas:
            for stock in self.position_list:
                sec_data = self.getdatabyname(stock)
                
                position_size = self.broker.getposition(data=sec_data).size
                if position_size > 0:
                    self.log('try sell:' + sec_data._name)
                    self.sell(sec_data,size=position_size)
                    #self.close(data)
            self.position_list = []

            # 获取计算的因子
            result_list=[]
            for stock in index_50_list:
                data=self.getdatabyname(stock)
                #remove un-aligned data
                if (data.datetime.date(0) != current_date):
                    continue

                close_list = data.close.get(size=self.p.look_back_days)
                # 对数据清洗的时候，默认缺失值等于0.0000001,所以，要去除缺失值部分
                close_list=[ i for i in close_list if i>0.001]
                rate=bt.mathsupport.standarddev(close_list)
                result_list.append([stock,rate])

            assert len(result_list)>0
            # 计算是否有新的股票并进行开仓    
            # long_list=[]
            # short_list=[]
            # self.log("index_300_list:{}".format(index_300_list))
            # 新调入的股票做多
            sorted_result_list=sorted(result_list,key=lambda x:x[1])
            i = self.p.std 
            long_list=[i[0] for i in sorted_result_list[self.p.portfolio_size*(i-1) :
                                                        self.p.portfolio_size*i]]
            assert len(long_list)>0
            # long_list=[i[0] for i in sorted_result_list[-10:]]

            # 分析当前持有的股票，看是否在long_list中，如果不在，就平仓
            close_num = 0
            """
            for stock in position_list:
                if stock not in long_list:
                    data = self.getdatabyname(stock)
                    self.close(data)
                    close_num+=1
            """

            # 分析当前要持有的股票，在position_list中是否存在
            for stock in long_list:
                if stock not in position_list:
                    data = self.getdatabyname(stock)
                    # 计算理论上应该买入的股票数目
                    account_value = self.broker.getvalue()
                    account_cash  = self.broker.getcash()
                    if close_num == 0:
                        every_stock_cash = account_cash/self.p.portfolio_size
                        stock_num = int(0.01*every_stock_cash/data.close[0])*100
                        if stock_num > 0:
                            self.buy(data, size=stock_num, exectype=bt.Order.Market)
                                     #valid=bt.Order.DAY)
                            self.position_list.append(data._name)
                    if close_num >= 1:
                        every_stock_cash = account_cash/close_num 
                        stock_num = int(0.01*every_stock_cash/data.close[0])*100
                        if stock_num > 0:
                            self.buy(data, size=stock_num, exectype=bt.Order.Market)
                                     #valid=bt.Order.DAY)
                            self.position_list.append(data._name)
                        
            self.log('end change position', current_date)

        
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
    ticker_list = ["AAPL", "FB", "AMZN", "MSFT", "CSCO"]
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
    
    #cerebro.broker.setcommission(commission=5,margin=2000.0) #stocklike=True)
    cerebro.broker.setcommission(commission=0.0001,stocklike=True)
    cerebro.broker.setcash(100000.0)
    cerebro.addstrategy(momentum_factor_strategy,
                        ticker_list, dataId_to_ticker_dic, ticker_to_dataId_dic)
                        #end_date.strftime("%Y-%m-%d"))
    # 添加相应的费用，杠杆率
    # 获取策略运行的指标
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    # cerebro.plot()
    end_time=time.time()
    print("total running time:{}".format(end_time-begin_time))
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

def group_plot():
    file_list = [] #os.listdir("/home/yun/Documents/")
    df = pd.DataFrame(index=range(3623))
    for file in file_list:
        if 'value' in file:
            df0 = pd.read_csv("/home/yun/Documents/"+file,index_col=0)
            std=file.split('__')[-1].split('_')[0]
            df.index=pd.to_datetime(df0.index)
            df[std]=df0.total_value
    df=df[['1','2','3','4','5']]
    df.plot()


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
