from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
#import os
#import sys  
#import numpy as np
#import random
#import plotly as py
#import plotly.graph_objs as go
import pandas as pd
import backtrader as bt

#from backtrader.plot.plot import plot_results  #我自己编写的，运行时去掉
from backtrader.feeds import GenericCSVData
"""
date 	        close 	        high 	        low 	        open 	        volume 	        money 	month_high_price
2002-01-04 	1316.4600 	1316.4600 	1316.4600 	1316.4600 	0.000000e+00 	0.000000e+00 	1316.4600
2002-01-07 	1302.0800 	1302.0800 	1302.0800 	1302.0800 	0.000000e+00 	0.000000e+00 	1316.4600
2002-01-08 	1292.7100 	1292.7100 	1292.7100 	1292.7100 	0.000000e+00 	0.000000e+00 	1316.4600
"""

# 在交易信息之外，额外增加了PE、PB指标，做其他策略使用
class GenericCSV_PB_PE(GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    #lines = ('month_high_price',)

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('month_high_price', 5),)

class aip_strategy(bt.Strategy):
    params = ()

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.count=0
        self.pre_month=None
    def prenext(self):
        pass 


    def next(self):
        # 假设有100万资金，每次成份股调整，每个股票使用1万元
        self.count+=1
        
        # 得到当天的时间
        current_date=self.datas[0].datetime.date(0)
        # check if the first trading date
        if self.pre_month is None or str(current_date)[:-3]>self.pre_month:
            #if self.datas[0].close[0]==self.datas[0].month_high_price[0]:
                lots=fund_per_size/self.datas[0].close[0]
                self.buy(size=lots)
                self.pre_month=str(current_date)[:-3]
        
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                

    def notify_trade(self, trade):
        if trade.isclosed:
            
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

begin_time=time.time()
fund_per_size = 1000
cerebro = bt.Cerebro()
# cerebro.broker = bt.brokers.BackBroker()  # 0.5%
# cerebro.broker.set_slippage_fixed(1, slip_open=True)
# Add a strategy
cerebro.addstrategy(aip_strategy)
    # 优化参数
    #strats = cerebro.optstrategy(
    #    momentum_strategy,
    #    look_back_days=[5,10],
    #    hold_days=[5,10,15,20],
    #    hold_percent=[0.1,0.2])
    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
kwargs = dict(fromdate = datetime.datetime(2010,1, 4),
            todate = datetime.datetime(2019, 11, 29),
            timeframe = bt.TimeFrame.Days,
            compression = 1,
            dtformat=('%Y-%m-%d'),
            datetime=0,
            high=2,
            low=3,
            open=1,
            close=4,
            volume=6,
            openinterest=-1,
            month_high_price=5,
            )

"""
use kwargs to reorder the data format from default of GenericCSVData:
CSV data format: Date,Open,High,Low,Close,Adj Close,Volume
class GenericCSVData(feed.CSVDataBase)
    params = (
        ('nullvalue', float('NaN')),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('tmformat', '%H:%M:%S'),
        ('datetime', 0),
        ('time', -1),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
    )
"""

feed = GenericCSV_PB_PE(dataname="/home/jun/proj/bt/data/backtest/SPY_aip.csv", **kwargs)
cerebro.adddata(feed, name = "spy")
cerebro.broker.setcommission(commission=0.0005)
cerebro.broker.setcash(100000.0)
# 添加相应的费用，杠杆率
# 获取策略运行的指标
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
cerebro.addanalyzer(bt.analyzers.Calmar, _name='_Calmar')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='_TimeDrawDown')
cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='_GrossLeverage')
cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='_PositionsValue')
cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='_LogReturnsRolling')
cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='_PeriodStats')
cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')
cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
cerebro.addanalyzer(bt.analyzers.SQN, _name='_SQN')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
cerebro.addanalyzer(bt.analyzers.Transactions, _name='_Transactions')
cerebro.addanalyzer(bt.analyzers.VWR, _name='_VWR')
results = cerebro.run()
cerebro.plot()
#plot_results(results,"/home/yun/沪深300悲剧先生投资策略.html")
print('End Portfolio Value: %.2f' % cerebro.broker.getvalue())
end_time=time.time()
print("total time:{}".format(end_time-begin_time))






