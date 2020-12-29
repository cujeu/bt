from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class MyStochastic1(bt.Indicator):
    lines = ('k', 'd', 'mystoc',)  # declare the output lines
    params = (
        ('k_period', 14),  # lookback period for highest/lowest
        ('d_period', 3),  # smoothing period for d with the SMA
    )

    def __init__(self):
        # declare the highest/lowest
        highest = bt.ind.Highest(self.data, period=self.p.k_period)
        lowest = bt.ind.Lowest(self.data, period=self.p.k_period)
        # calculate and assign lines
        self.lines.k = k = (self.data - lowest) / (highest - lowest)
        self.lines.d = d = bt.ind.SMA(k, period=self.p.d_period)
        self.lines.mystoc = abs(k - k(-1)) / 2.0

class MyStochastic2(bt.Indicator):
    lines = ('k', 'd', 'mystoc',)
    # manually counted period
    # 14 for the fast moving k
    # 3 for the slow moving d
    # No extra for the previous k (-1) is needed because
    # already buffers more than the 1 period lookback
    # If we were doing d - d(-1), there is nothing making
    # sure k(-1) is being buffered and an extra 1 would be needed
    params = (
        ('k_period', 14),  # lookback period for highest/lowest
        ('d_period', 3),  # smoothing period for d with the SMA
    )
    def __init__(self):
        self.addminperiod(self.p.k_period + self.p.d_period)

    def next(self):
        # Get enough data points to calculate k and do it
        d = self.data.get(size=self.p.k_period)
        hi = max(d)
        lo = min(d)
        self.lines.k[0] = k0 = (self.data[0] - lo) / (hi - lo)
        # Get enough ks to calculate the SMA of k. Assign to d
        last_ks = self.l.k.get(size=self.p.d_period)
        self.lines.d[0] = sum(last_ks) / self.p.d_period
        # Now calculate mystoc
        self.lines.mystoc[0] = abs(k0 - self.l.k[-1]) / 2.0

class MyTrix(bt.Indicator):

    lines = ('trix',)
    params = (('period', 15),)

    def __init__(self):
        ema1 = btind.EMA(self.data, period=self.p.period)
        ema2 = btind.EMA(ema1, period=self.p.period)
        ema3 = btind.EMA(ema2, period=self.p.period)

        self.lines.trix = 100.0 * (ema3 - ema3(-1)) / ema3(-1)

class PriceDiv(bt.Indicator):
    lines = ('cs', 'sm', 'ml')
    params = (('shortPeriod', 20),
              ('midPeriod', 60),
              ('longPeriod', 120),)

    def __init__(self):
        #ema20 = btind.ExponentialMovingAverage(self.data, period=self.p.shortPeriod)
        #ema60 = btind.ExponentialMovingAverage(self.data, period=self.p.midPeriod)

        ema20 = btind.EMA(self.data, period=self.p.shortPeriod)
        ema60 = btind.EMA(self.data, period=self.p.midPeriod)
        ema120 = btind.EMA(self.data, period=self.p.longPeriod)
        self.lines.cs = ((self.data - ema20) / ema20) * 100.0
        self.lines.sm = ((ema20 - ema60) / ema60) * 100.0
        self.lines.ml = ((ema60 - ema120) / ema120) * 100.0

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
        #print(len(self.lines.stc))

class NoStrategy(bt.Strategy):
    params = (('trixperiod', 15),)

    def __init__(self):
        self.bar_num = 0
        #MyTrix(self.data, period=self.p.trixperiod)
        self.data_priceDiv = PriceDiv(self.data)
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
              self.data_schaff.stc[0], self.data_schaff.stc[-1],
              self.data_priceDiv.cs[0])
        #print(self.bar_num, len(self.data_schaff.lines[0]),
        #      self.data_schaff.lines[0].array[idx], self.data_schaff.lines[0].array[idx-1])
        

if __name__ == '__main__':
    today = datetime.datetime.today().date()
    shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
    end_date = today - shift + datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=5*365)-datetime.timedelta(days=1)
    #start_date = end_date - datetime.timedelta(weeks=4)-datetime.timedelta(days=1)
    print(today, shift, end_date, start_date)

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
