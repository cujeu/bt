# 按照标准动量策略做一个多因子回测的模板，开始测试各个单个因子能够带来超额alpha
# ref: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701346
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
import os
import sys  
import backtrader as bt
import numpy as np
import pandas as pd
import random
import pickle 
from backtrader.plot.plot import run_cerebro_plot  #我自己编写的，运行时去掉

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

# 编写策略
class momentum_factor_strategy(bt.Strategy):
    author = "yun"
    params = (("look_back_days",30),("hold_days",30),)

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.bar_num=0
        self.index_50_date_stock_dict=self.get_index_50_date_stock()

    

    def prenext(self):
        
        pass

    def next(self):
        # 假设有100万资金，每次成份股调整，每个股票使用1万元
        self.bar_num+=1
        # 需要调仓的时候
        if self.bar_num%self.p.hold_days==0:
            # 得到当天的时间
            current_date=self.datas[0].datetime.date(0)
            # 获得当天的交易的股票
            index_50_list=self.index_50_date_stock_dict[str(current_date)]
            # 先全部平仓
            for data in self.datas:
                position_size=self.broker.getposition(data=data).size
                if position_size!=0:
                    self.close(data)
            # 获取计算的因子
            result_list=[]
            for stock in index_50_list:
                data=self.getdatabyname(stock)
                if len(data)>self.p.look_back_days:
                    now_close=data.close[0]
                    pre_n_close=data.close[-self.p.look_back_days+1]
                    # 对数据清洗的时候，默认缺失值等于0.0000001
                    if pre_n_close>0.01:
                        rate=(now_close-pre_n_close)/pre_n_close
                        result_list.append([stock,rate])
            # 计算是否有新的股票并进行开仓    
            long_list=[]
            short_list=[]
            # self.log("index_300_list:{}".format(index_300_list))
            # 新调入的股票做多
            sorted_result_list=sorted(result_list,key=lambda x:x[1])
            short_list=[i[0] for i in sorted_result_list[:10]]
            long_list=[i[0] for i in sorted_result_list[-10:]]
            # 循环股票，决定做多和做空
            # 得到当前的账户价值
            total_value = self.broker.getvalue()
            every_stock_value = total_value/20
            for data in self.datas:
                if data._name in long_list:
                    close_price=data.close[0]
                    lots=int(every_stock_value/close_price)
                    self.sell(data,size=lots)
                
                if data._name in short_list:
                    close_price=data.close[0]
                    lots=int(every_stock_value/close_price)
                    self.buy(data,size=lots)
        
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

    def get_index_50_date_stock(self,data_path="/home/yun/data/index_50_date_stock_dict.pkl"):
        '''从外部添加每个交易日的上证50的成分股代码'''
        import pickle 
        with open(data_path,'rb') as f:
            data=pickle.load(f)
        return data 
begin_time=time.time()
cerebro = bt.Cerebro()
cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%
# cerebro.broker.set_slippage_fixed(1, slip_open=True)
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
cerebro.broker.setcommission(commission=0.0005,stocklike=True)
cerebro.broker.setcash(1000000.0)
# 添加相应的费用，杠杆率
# 获取策略运行的指标
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
# results = cerebro.run()
# plot_results(results,"/home/yun/沪深300悲剧先生投资策略.html")
run_cerebro_plot(cerebro,momentum_factor_strategy,{},result_path="/home/yun/Documents/")
end_time=time.time()
print("一共使用时间为:{}".format(end_time-begin_time))




