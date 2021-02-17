from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import datetime
import pandas as pd
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

        """ another squeeze indicator from trading view
        pairSQ(ma1,ma2,len) =>
            d = abs(ma1-ma2)
            change = (1-(d[0] / (d[0]+d[len])))*100
        """

class RelativeMovingAverage(bt.Indicator):
    '''
    A Moving Average that smoothes data exponentially over time.

    It is a subclass of SmoothingMovingAverage.

      - self.smfactor -> 1 / (1 + period)
      - self.smfactor1 -> `1 - self.smfactor`

    Formula:
      - movav = prev * (1.0 - smoothfactor) + newdata * smoothfactor

    See also:
      - http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    '''
    alias = ('RMA', )
    lines = ('rma',)
    params = (('period', 10),)

    def __init__(self):
        # Before super to ensure mixins (right-hand side in subclassing)
        # can see the assignment operation and operate on the line
        self.lines[0] = es = bt.indicators.ExponentialSmoothing(
            self.data,
            period=self.p.period,
            alpha=1.0 / (1.0 + self.p.period))

        self.alpha, self.alpha1 = es.alpha, es.alpha1

        super(RelativeMovingAverage, self).__init__()

class AtrTrend(bt.Indicator):
    lines = ('up', 'dn',)
    params = (('ATRPeriod', 10),
              ('Multiplier', 3.0),)

    def __init__(self):

        hl2 = (self.data.high + self.data.low) / 2
        #ttr = bt.indicators.TrueRange(self.data)
        #ttr = bt.Max(self.data.high- self.data.low, 
        #             bt.Max(abs(self.data.high(0) - self.data.close(-1)),
        #                    abs(self.data.low(0) - self.data.close(-1))))

        #atr2 = RelativeMovingAverage(ttr, period=self.p.ATRPeriod)
        #atr30 = bt.ind.SMA(ttr, period=30)
        #atr20 = bt.ind.SMA(ttr, period=20)
        #atr2 = atr30 - atr20 + bt.ind.SMA(ttr, period=self.p.ATRPeriod)
        #atr2 = bt.ind.EMA(ttr, period=self.p.ATRPeriod)
        atr2 = bt.indicators.AverageTrueRange(self.data, period=self.p.ATRPeriod)
        upSrc = hl2 - (self.p.Multiplier * atr2)
        #up := close[1] > up1 ? max(up,up1) : up
        self.lines.up = btind.If(self.data.close(-1) > upSrc(-1),
                                 bt.Max(upSrc,upSrc(-1)) , upSrc)
        
        dnSrc = hl2 + (self.p.Multiplier * atr2)
        #dn := close[1] < dn1 ? min(dn, dn1) : dn
        self.lines.dn = btind.If(self.data.close(-1) < dnSrc(-1),
                                 bt.Min(dnSrc, dnSrc(-1)) , dnSrc)
        
        """
        trd = self.data.close - self.data.close + 1
        trd2 = btind.If(trd(-1) < 0,
                        btind.If(self.data.close > dn,1,trd2(-1)),
                        btind.If(self.data.close < up, -1, trd2(-1)))
        #trd2 = btind.If((trd(-1) < 0) and (self.data.close(0) > dn(0)),
        #1, btind.If(trd(-1) > 0 and self.data.close(0) < up(0), -1, 0))

        self.lines.trend = trd2
        #btind.If((trd(-1) < 0) and (self.data.close > dn),
        #               1,  btind.If(trd(-1) > 0 and self.data.close < up, -1, 0))
        """
        super(AtrTrend, self).__init__()

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

class OnBalanceVolume(bt.Indicator):
    lines = ('obv',)
    params = (('period', 1),)
    #plotinfo = dict(subplot=False)  # plot in same axis as the data
    
    def __init__(self):
        upday = btind.UpDay(self.data.close)
        downday = btind.DownDay(self.data.close)

        adup = btind.If(upday, self.data.volume, 0.0)
        addown = btind.If(downday, 0-self.data.volume, 0.0)

        self.lines.obv = btind.Accum(adup + addown)

        super(OnBalanceVolume, self).__init__()

class KeltnerChannel(bt.Indicator):
    alias = ('KCBands',)
    lines = ('top', 'mid', 'bot',)
    params = (('periodEMA', 20), ('periodATR', 20),('devfactor', 2.0),)
    plotinfo = dict(subplot=False)
    plotlines = dict(
        mid=dict(ls='--'),
        top=dict(_samecolor=True),
        bot=dict(_samecolor=True),
    )

    def __init__(self):
        #        self.addminperiod(self.p.period)
        self.lines.mid = bt.indicators.EMA(self.data, period=self.p.periodEMA)
        atr = bt.indicators.AverageTrueRange(self.data, period=self.p.periodATR)
        self.lines.top = self.lines.mid + self.p.devfactor * atr
        self.lines.bot = self.lines.mid - self.p.devfactor * atr
        super(KeltnerChannel, self).__init__()

class RGChannel(bt.Indicator):
    alias = ('RGBands',)
    lines = ('shortRG', 'midRG', 'longRG',)
    params = (('periodShort', 20), ('periodMid', 60),('periodLong', 120),)
    plotinfo = dict(subplot=False)
    plotlines = dict(
        shortRG=dict(ls='--'),
        midRG=dict(_samecolor=True),
        longRG=dict(_samecolor=True),
    )

    def __init__(self):
        #        self.addminperiod(self.p.period)
        h = bt.indicators.Highest(self.data.high, period=self.p.periodShort)
        l = bt.indicators.Lowest(self.data.low, period=self.p.periodShort)
        self.lines.shortRG = (h - l) * 100 / l
        
        h = bt.indicators.Highest(self.data.high, period=self.p.periodMid)
        l = bt.indicators.Lowest(self.data.low, period=self.p.periodMid)
        self.lines.midRG = (h - l) * 100 / l
        
        h = bt.indicators.Highest(self.data.high, period=self.p.periodLong)
        l = bt.indicators.Lowest(self.data.low, period=self.p.periodLong)
        self.lines.longRG = (h - l) * 100 / l
        
        super(RGChannel, self).__init__()

class NoStrategy_with_stc(bt.Strategy):
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
        
class NoStrategy_with_obv(bt.Strategy):
    params = (('trixperiod', 15),)

    def __init__(self):
        self.bar_num = 0
        #MyTrix(self.data, period=self.p.trixperiod)
        #self.data_priceDiv = PriceDiv(self.data)
        self.data_obv = OnBalanceVolume(self.data)  ##
        ##btind.obv(self.data,period=3, plot=True)
        self.data_obv20 = btind.SMA(self.data_obv, period=20)


    def next(self):
        self.bar_num += 1
        if self.bar_num < 50 :
            return
        #print("div ",self.data_priceDiv.cs[0],
        print(self.data.close[0],
              "obv ",self.data_obv[0],
              "obv20 ",self.data_obv20[0])

class NoStrategy_with_band(bt.Strategy):

    def __init__(self):
        self.bar_num = 0
        self.data_rg = RGChannel(self.data)
        #self.data_bb = btind.BollingerBands(self.data, period=14, devfactor=4)
        #self.data_bb = KeltnerChannel(self.data)


    def next(self):
        self.bar_num += 1
        if self.bar_num < 50 :
            return
        #print("div ",self.data_priceDiv.cs[0],
        print(self.data.close[0],
              "top ",self.data_rg.shortRG[0],
              "mid ",self.data_rg.midRG[0],
              "bot ",self.data_rg.longRG[0])
        """
        print(self.data.close[0],
              "top ",self.data_bb.top[0],
              "mid ",self.data_bb.mid[0],
              "bot ",self.data_bb.bot[0])
        """

class NoStrategy(bt.Strategy):

    def __init__(self):
        self.bar_num = 0
        #ttr = bt.indicators.TrueRange(self.data)
        #self.data_atr = RelativeMovingAverage(bt.indicators.TrueRange(self.data), period=10)
        #atr3 = btind.AverageTrueRange(self.data, period=30)
        #atr2 = btind.AverageTrueRange(self.data, period=20)
        #atr1 = btind.AverageTrueRange(self.data, period=10)
        #self.data_atr = atr3 - atr2 + atr1
        
        #ttr = bt.Max(self.data.high- self.data.low, 
        #             bt.Max(abs(self.data.high(0) - self.data.close(-1)),
        #                    abs(self.data.low(0) - self.data.close(-1))))
        #self.data_atr = RelativeMovingAverage(ttr, period=10)

        self.data_atr = btind.AverageTrueRange(self.data, period=10)
        self.data_trend = AtrTrend(self.data)
        self.trend = -1

    def next(self):
        self.bar_num += 1
        if self.bar_num < 50 :
            return
        #trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        mindn = min(self.data_trend.dn[-5], min(self.data_trend.dn[-6], self.data_trend.dn[-2]))
        maxup = max(self.data_trend.up[-5], max(self.data_trend.up[-6], self.data_trend.up[-2]))

        #maxup = self.data_trend.up[-1]
        #mindn = self.data_trend.dn[-1]
        if (self.trend > 0 and self.data.close[0] < maxup): #self.data_trend.up[-2]) :
            self.trend = -1
        elif (self.trend < 0 and self.data.close[0] > mindn ):  #self.data_trend.dn[-2]):
            self.trend = 1
        #print("div ",self.data_priceDiv.cs[0],
        print(str(self.datetime.date(ago=0)), 
              self.data.close[0],
              (self.data.high[0] + self.data.low[0])/2,
              "trend ", self.trend, self.data_atr[0], self.data_trend.up[0], self.data_trend.dn[0])


if __name__ == '__main__':
    run_test = False
    if run_test:
        today = datetime.datetime.today().date()
        shift = datetime.timedelta(max(1,(today.weekday() + 6) % 7 - 3))
        end_date = today - shift + datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=2*365)-datetime.timedelta(days=1)
        #start_date = end_date - datetime.timedelta(weeks=4)-datetime.timedelta(days=1)
        print(today, shift, end_date, start_date)
    
        # Create a cerebro entity
        cerebro = bt.Cerebro()
    
        # Add a strategy
        cerebro.addstrategy(NoStrategy)
    
        # Create a Data Feed
        #datapath = ('/home/jun/proj/backtrader/datas/2006-day-001.txt')
        datapath = ('SPY20.csv')
        trading_data_df = pd.read_csv(datapath, index_col=0, parse_dates=True)
        trading_data_df.drop(['Adj Close'], axis=1, inplace=True)
        trading_data_df.index.names = ['date']
        trading_data_df.rename(columns={'Open' : 'open', 'High' : 'high', 'Low' : 'low',
                                        'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
        ## set data range by date
        # Delete these row indexes from dataFrame
        ##indexDates = trading_data_df[trading_data_df.index < start_date].index
        ##        trading_data_df.drop(indexDates , inplace=True)
        trading_data_df['openinterest'] = 0
        data_feed = bt.feeds.PandasData(dataname=trading_data_df)
                                                #fromdate = datetime.datetime(2010, 1, 4),
                                                #todate = datetime.datetime(2019, 12, 31))
                                                #fromdate=start_date,
                                                #todate=end_date)  # dtformat=('%Y-%m-%d'))
        
        """
        data = bt.feeds.BacktraderCSVData(dataname=datapath,
                                            datetime=0,
                                            high=1,
                                            low=2,
                                            open=3,
                                            close=4,
                                            adjclose=5,
                                            volume=6,
                                            openinterest=-1)
        """
    
        # Add the Data Feed to Cerebro
        cerebro.adddata(data_feed)
    
        # Run over everything
        cerebro.run()
        print(today, shift, end_date, start_date)
    
        # Plot the result
        cerebro.plot()
