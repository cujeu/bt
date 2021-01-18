# price divergence factor for ETF poll
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
import os
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
        self.position_list = [] #postion order ["sym", "date", "close", "high", "perc", "exit","edate","eperc"]
        self.csCount = 0

        self.hasPosition = False
        self.entryPrice = 100
        self.highPrice = 100
        self.newLowDiv = 100
        self.newLowBar = 0
        self.atrTrend = False
        self.cond_div = False
        self.cond_atr = False

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
            #d.data_bb = btind.BollingerBands(d, period=14, devfactor=2.0)
            #d.data_kc = KeltnerChannel(d, periodEMA = 20, periodATR = 20, devfactor=1.5 )
            #d.data_rg = RGChannel(d)

            self.log('stategy data init :' + d._name)

    def stop(self):
        if self.hasPosition:
            stock = self.pool_list[0]
            data = self.getdatabyname(stock)
            self.hasPosition = False
            result_list = self.position_list[-1]
            del self.position_list[-1]
                # ["sym", "date", "close", "high", "perc", "exit","edate","eperc"]
            result_list[3] = str(round(self.highPrice,2))
            result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
            result_list[5] = str(round(data.close[0],2))
            result_list[6] = str(self.current_date)
            result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
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
            
            if self.cond_div:
                if (data.priceDiv.cs[0] < 0 and data.priceDiv.cs[-1] > 0 and \
                    data.priceDiv.cs[0] < data.priceDiv.sm[0]):
                    self.cond_div = False
                    self.hasPosition = False
                    result_list = self.position_list[-1]
                    del self.position_list[-1]
                    # ["sym", "date", "close", "high", "perc", "exit","edate","eperc"]
                    result_list[3] = str(round(self.highPrice, 2))
                    result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
                    result_list[5] = str(round(data.close[0], 2))
                    result_list[6] = str(self.current_date)
                    result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
                    self.position_list.append(result_list)

            elif not self.atrTrend and self.cond_atr:
                self.cond_atr = False
                self.hasPosition = False
                result_list = self.position_list[-1]
                del self.position_list[-1]
                # ["sym", "date", "close", "high", "perc", "exit","edate","eperc"]
                result_list[3] = str(round(self.highPrice, 2))
                result_list[4] = str(round(100 * (self.highPrice - self.entryPrice) / self.entryPrice, 0))
                result_list[5] = str(round(data.close[0], 2))
                result_list[6] = str(self.current_date)
                result_list[7] = str(round(100 * (data.close[0] - self.entryPrice) / self.entryPrice, 0))
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
                
            """
                COND_1      = if  close > ema_mm and OBV > OBVMA20 then 1 else 0;
                COND_2      = if cs < 10 and sm < 15 and ml < 20 then 1 else 0;
                COND_3      = if  RG_ss < 30 and RG_mm < 45 and RG_ll < 60 then 1 else 0;
                COND_4      = if DEA > 0 and MACD_RATE > MACD_RATE[1] then 1 else 0;
                COND_5      = SQUEEZE_OFF;
            """
            #cond_1 = (data.close[0] > data.ema60[0] and data.close[-1] < data.ema60[-1] and data.data_obv[0] > data.data_obv20[0])
            #old conservation div condition
            #cond_div = (data.priceDiv.cs[0] < -7  and
            #            data.priceDiv.cs[0] > data.priceDiv.cs[-1])
            
            if not self.cond_div:
                self.cond_div = ((data.priceDiv.cs[0] < -10 or
                                 (data.priceDiv.cs[0] < -5 and 
                                  data.priceDiv.cs[0] > self.newLowDiv and
                                  (self.bar_num - self.newLowBar) > 1)) and
                                 data.priceDiv.cs[0] > data.priceDiv.cs[-1])

            if not self.cond_atr:
                self.cond_atr = self.atrTrend

            """
            cond_1 = (data.close[0] > data.ema60[0] and data.data_obv[0] > data.data_obv20[0])
            cond_2 = (data.priceDiv.cs[0] > data.priceDiv.cs[-1] and data.priceDiv.cs[0] < 10 and data.priceDiv.sm[0] < 15 and data.priceDiv.cs[0] < data.priceDiv.sm[0])
            cond_3 = (data.data_rg.shortRG[0] < 30 and data.data_rg.midRG[0] < 45 and data.data_rg.longRG[0] < 60)
            cond_4 = (data.macd.signal[0] > 0 and data.macd.signal[0] > data.macd.signal[-1])
            cond_5 = (data.data_bb.bot[0] < data.data_kc.bot[0] and data.data_bb.top[0] > data.data_kc.top[0])
            if (cond_1 and cond_2 and cond_3 and cond_4 and cond_5) or cond_div:
            """
            #if (cond_atr and (data.priceDiv.cs[0] > data.priceDiv.sm[0] and
            #                 data.priceDiv.cs[0] < data.priceDiv.ml[0] and
            #                 data.priceDiv.sm[0] > data.priceDiv.sm[-1])) or 

            if self.cond_atr or self.cond_div :
                            
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

        mindn = min(self.data.data_atr.dn[-5], min(self.data.data_atr.dn[-3], self.data.data_atr.dn[-2]))
        maxup = max(self.data.data_atr.up[-5], max(self.data.data_atr.up[-3], self.data.data_atr.up[-2]))
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
                    g_alert_list.append([to_watch_list[0],str(self.current_date)])
                    self.log('recent position:' + to_watch_list[0] + ' '+str(self.current_date))


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
    mk_df = pd.DataFrame(columns = ["sym", "date", "close", "high", "perc", "exit","edate","eperc"])
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
            # e is list of ["sym", "date", "close", "high", "perc", "exit","edate","eperc"]
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

    
def runstrat(args=None):
    today = datetime.datetime.today().date()
    shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
    end_date = today - shift + datetime.timedelta(days=1)
    #start_date = datetime.datetime(2015, 1,1)
    start_date = end_date - datetime.timedelta(days=4*365)
    # ticker_list[0] must cover start_date to end_date, as a reference

    ## 1. scan ETF
    ticker_list = get_etf_symbols()
    #ticker_list = ['TECL','FNGU','FNGO','CWEB','TQQQ','ARKW','ARKG','ARKK','QLD' ,'ROM']
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
    mk_df = start_scan(ticker_list, start_date, end_date)
    filename = conf_data_path + 'scan_ark.csv'
    mk_df.to_csv(filename, encoding='utf8')
    #with open(filename, 'a') as f:
    #    mk_df.to_csv(f, header=False)
    #    f.close

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
 
if __name__ == '__main__':
    runstrat()
