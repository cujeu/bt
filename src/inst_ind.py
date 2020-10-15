from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class MyTrix(bt.Indicator):

    lines = ('trix',)
    params = (('period', 15),)

    def __init__(self):
        ema1 = btind.EMA(self.data, period=self.p.period)
        ema2 = btind.EMA(ema1, period=self.p.period)
        ema3 = btind.EMA(ema2, period=self.p.period)

        self.lines.trix = 100.0 * (ema3 - ema3(-1)) / ema3(-1)

class SchaffTrend(bt.Indicator):
    lines = ('stc',)
    params = (
        ('fastPeriod', 23),
        ('slowPeriod', 50),
        ('kPeriod', 10),
        ('dPeriod', 3),
        ('movav', btind.MovAv.Exponential) )

    def __init__(self):
        stc_macd = self.p.movav(self.data, period=self.p.fastPeriod) - self.p.movav(self.data, period=self.p.slowPeriod)
        high = btind.Highest(stc_macd, period=self.p.kPeriod)
        low = btind.Lowest(stc_macd, period=self.p.kPeriod)
        fastk1= btind.If(high-low > 0, (stc_macd-low) / (high-low) * 100, 0)
        fastk1= btind.If(high-low > 0, (stc_macd-low) / (high-low) * 100, fastk1(-1))
        fastd1 = self.p.movav(fastk1, period=self.p.dPeriod)

        high2 = btind.Highest(fastd1, period=self.p.kPeriod)
        low2 = btind.Lowest(fastd1, period=self.p.kPeriod)
        fastk2 = btind.If(high2-low2 > 0, (fastd1(0)-low2) / (high2-low2) * 100, 0)
        fastk2 = btind.If(high2-low2 > 0, (fastd1(0)-low2) / (high2-low2) * 100, fastk2(-1))
        self.lines.stc = self.p.movav(fastk2, period=self.p.dPeriod)
        print(len(self.lines.stc))

class NoStrategy(bt.Strategy):
    params = (('trixperiod', 15),)

    def __init__(self):
        self.bar_num = 0
        MyTrix(self.data, period=self.p.trixperiod)
        self.data_schaff = SchaffTrend(self.data)
        #print(self.data.schaff.lines.stc)


    def next(self):
        self.bar_num += 1
        if self.bar_num < 50 :
            return
        # if data.schaff > 50: 
        #print(self.bar_num, len(self.data_schaff.lines.stc), self.data_schaff.lines.stc[self.bar_num-1])
        idx = len(self.data_schaff.lines[0]) - 1
        print(self.bar_num, len(self.data_schaff.lines[0]),
              self.data_schaff.stc[0], self.data_schaff.stc[-1])
        #print(self.bar_num, len(self.data_schaff.lines[0]),
        #      self.data_schaff.lines[0].array[idx], self.data_schaff.lines[0].array[idx-1])
        

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(NoStrategy, trixperiod=15)

    # Create a Data Feed
    #datapath = ('/home/jun/proj/backtrader/datas/2006-day-001.txt')
    datapath = ('SPY20.csv')
    data = bt.feeds.BacktraderCSVData(dataname=datapath)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Run over everything
    cerebro.run()

    # Plot the result
    cerebro.plot()
