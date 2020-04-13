"""
Indicators as shown by Peter Bakker at:
https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
"""

"""
25-Mar-2018: Fixed syntax to support the newest version of Pandas. Warnings should no longer appear.
             Fixed some bugs regarding min_periods and NaN.
			 If you find any bugs, please report to github.com/palmbook
"""

# Import Built-Ins
# import logging

# Import Third-Party
import pandas as pd
import numpy as np
from config import *

# Import Homebrew

# Init Logging Facilities
# log = logging.getLogger(__name__)


def moving_average(df, n, ma_name, column_name = None):
    """Calculate the moving average for the given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    MA = pd.Series(df[column_name].rolling(n, min_periods=n).mean(), name=ma_name)
    df = df.join(MA)
    df.fillna(method='bfill', inplace=True )  #fill nan with next valid value
    return df


def exponential_moving_average(df, n, ema_name, column_name = None):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close
    
    # ema_name = 'EMA_' + str(n)
    EMA = pd.Series(df[column_name].ewm(span=n, min_periods=n).mean(), name=ema_name)
    df = df.join(EMA)
    return df


def momentum(df, n, mom_name, column_name = None):
    """
    
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    # mom_name='Momentum_' + str(n)
    M = pd.Series(df[column_name].diff(n), name=mom_name)
    df = df.join(M)
    return df


def rate_of_change(df, n, roc_name, column_name = None):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close
    M = df[column_name].diff(n - 1)
    N = df[column_name].shift(n - 1)
    # roc_name='ROC_' + str(n)
    ROC = pd.Series(M / N, name=roc_name)
    df = df.join(ROC)
    return df


def average_true_range(df, n, atr_name, column_name = None):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0.0]
    if column_name is None:
        column_name = conf_close
    # print(df.loc[df.index[i + 1], conf_low])

    #df.loc[df.index[row_number], pd.IndexSlice[:, 'px']]
    while i < (len(df.index)-1):
        TR = max(df.loc[df.index[i + 1], conf_high], df.loc[df.index[i], column_name])
        TR = TR - min(df.loc[df.index[i + 1], conf_low], df.loc[df.index[i], column_name])
        TR_l.append(TR)
        i = i + 1
    ATR = pd.Series(TR_l).ewm(span=n, min_periods=n).mean()
    # atr_name = 'ATR_' + str(n)
    df[atr_name] = ATR.values
    #df.insert(loc=0, column=nm, value= ATR)
    #df = df.join(ATR, on = 'open_price') #, how='left', right_on = ['ATR_' + str(n)]) 
    return df


def bollinger_bands_v2(df, n, dist = None, column_name = None):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close
    if dist is None:
        dist = 2.0

    MA = pd.Series(df[column_name].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df[column_name].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df[column_name] - MA + dist * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

def bollinger_bands(df, n, dist = None, column_name = None):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close
    if dist is None:
        dist = 2.0

    MA = pd.Series(df[column_name].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df[column_name].rolling(n, min_periods=n).std())
    b1 = ( MA + dist * MSD)
    B1 = pd.Series(b1, name='BBUpper_' + str(n))
    df = df.join(B1)
    b2 = ( MA - dist * MSD) 
    B2 = pd.Series(b2, name='BBLower_' + str(n))
    df = df.join(B2)
    return df


def ppsr(df, column_name = None):
    """Calculate Pivot Points, Supports and Resistances for given data
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    PP = pd.Series((df[conf_high] + df[conf_low] + df[column_name]) / 3)
    R1 = pd.Series(2 * PP - df[conf_low])
    S1 = pd.Series(2 * PP - df[conf_high])
    R2 = pd.Series(PP + df[conf_high] - df[conf_low])
    S2 = pd.Series(PP - df[conf_high] + df[conf_low])
    R3 = pd.Series(df[conf_high] + 2 * (PP - df[conf_low]))
    S3 = pd.Series(df[conf_low] - 2 * (df[conf_high] - PP))
    psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
    PSR = pd.DataFrame(psr)
    ##print(df.iloc[[2287]])  ##print a row by index
    df = df.join(PSR)
    return df


def stochastic_oscillator_k(df, stochK_name, column_name = None):
    """Calculate stochastic oscillator %K for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    #stochK_name=stochK
    SOk = pd.Series((df[column_name] - df[conf_low]) / (df[conf_high] - df[conf_low]), name = stochK_name)
    df = df.join(SOk)
    return df


def stochastic_oscillator_d(df, n, stochD_name, column_name = None):
    """Calculate stochastic oscillator %D for given data.
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    SOk = pd.Series((df[column_name] - df[conf_low]) / (df[conf_high] - df[conf_low])) #, name='SO%k')
    # stochD_name = 'stochD' + str(n)
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name = stochD_name)
    df = df.join(SOd)
    return df


def trix(df, n, trix_name, column_name = None):
    """Calculate TRIX for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    EX1 = df[column_name].ewm(span=n, min_periods=n).mean()
    EX2 = EX1.ewm(span=n, min_periods=n).mean()
    EX3 = EX2.ewm(span=n, min_periods=n).mean()
    i = 0
    ROC_l = [np.nan]
    while i < (len(df.index)-1):
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    ##Trix = pd.Series(ROC_l, name='Trix_' + str(n))
    ##df = df.join(Trix)
    # trix_name = 'Trix_' + str(n)
    df[trix_name] = pd.Series(ROC_l).values
    return df


def average_directional_movement_index(df, n, n_ADX, adx_name, column_name = None):
    """Calculate the Average Directional Movement Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :param n_ADX: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    UpI = [0]
    DoI = [0]
    while i < (len(df.index)-1):
        UpMove = df.loc[df.index[i + 1], conf_high] - df.loc[df.index[i], conf_high]
        DoMove = df.loc[df.index[i + 1], conf_low] - df.loc[df.index[i], conf_low]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    
    i = 0
    TR_l = [0]
    while i < (len(df.index)-1):
        TR = max(df.loc[df.index[i + 1], conf_high], df.loc[df.index[i], column_name]);
        TR = TR - min(df.loc[df.index[i + 1], conf_low], df.loc[df.index[i], column_name])
        TR_l.append(TR)
        i = i + 1

    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() / ATR)
    ##ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean(),
    ##                name='ADX_' + str(n) + '_' + str(n_ADX))
    ##df = df.join(ADX)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean())
    # adx_name = 'ADX_' + str(n) + '_' + str(n_ADX)
    df[adx_name] = ADX.values
    return df


def macd(df, n_fast, n_slow, column_name = None):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    EMAfast = pd.Series(df[column_name].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df[column_name].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def mass_index(df, mass_index_name):
    """Calculate the Mass Index for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    Range = df[conf_high] - df[conf_low]
    EX1 = Range.ewm(span=9, min_periods=9).mean()
    EX2 = EX1.ewm(span=9, min_periods=9).mean()
    Mass = EX1 / EX2
    # mass_index_name='Mass Index')
    MassI = pd.Series(Mass.rolling(25).sum(), name = mass_index_name)
    df = df.join(MassI)
    return df


def vortex_indicator(df, n, vortex_name, column_name = None):
    """Calculate the Vortex Indicator for given data.
    
    Vortex Indicator described here:
        http://www.vortexindicator.com/VFX_VORTEX.PDF
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    TR = [0]
    while i < (len(df.index)-1):
        Range = max(df.loc[df.index[i + 1], conf_high], df.loc[df.index[i], column_name])
        Range = Range - min(df.loc[df.index[i + 1], conf_low], df.loc[df.index[i], column_name])
        TR.append(Range)
        i = i + 1

    i = 0
    VM = [0]
    while i < (len(df.index)-1):
        Range = abs(df.loc[df.index[i + 1], conf_high] - df.loc[df.index[i], conf_low])
        Range = Range - abs(df.loc[df.index[i + 1], conf_low] - df.loc[df.index[i], conf_high])
        VM.append(Range)
        i = i + 1
    ##VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum(), name='Vortex_' + str(n))
    ##df = df.join(VI)
    VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum())
    # vortex_name = 'Vortex_' + str(n)
    df[vortex_name] = VI.values
    return df


def kst_oscillator(df, r1, r2, r3, r4, n1, n2, n3, n4, column_name = None):
    """Calculate KST Oscillator for given data.
    
    :param df: pandas.DataFrame
    :param r1: 
    :param r2: 
    :param r3: 
    :param r4: 
    :param n1: 
    :param n2: 
    :param n3: 
    :param n4: 
    :return: pandas.DataFrame
    """
    M = df[column_name].diff(r1 - 1)
    N = df[column_name].shift(r1 - 1)
    ROC1 = M / N
    M = df[column_name].diff(r2 - 1)
    N = df[column_name].shift(r2 - 1)
    ROC2 = M / N
    M = df[column_name].diff(r3 - 1)
    N = df[column_name].shift(r3 - 1)
    ROC3 = M / N
    M = df[column_name].diff(r4 - 1)
    N = df[column_name].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(
        ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 + ROC3.rolling(n3).sum() * 3 + ROC4.rolling(n4).sum() * 4,
        name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
            n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df


def relative_strength_index(df, n, rsi_name, column_name = None):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    UpI = [0]
    DoI = [0]
    while i < (len(df.index)-1):
        UpMove = df.loc[df.index[i+1], conf_high] - df.loc[df.index[i], conf_high]
        DoMove = df.loc[df.index[i], conf_low] - df.loc[df.index[i+1], conf_low]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    ##RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    ##df = df.join(RSI)
    RSI = pd.Series(PosDI / (PosDI + NegDI))
    # rsi_name = 'RSI_' + str(n)
    df[rsi_name] = RSI.values
    return df


def true_strength_index(df, r, s, tsi_name, column_name = None):
    """Calculate True Strength Index (TSI) for given data.
    
    :param df: pandas.DataFrame
    :param r: 
    :param s: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    M = pd.Series(df[column_name].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span=r, min_periods=r).mean())
    aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r).mean())
    EMA2 = pd.Series(EMA1.ewm(span=s, min_periods=s).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s).mean())
    # tsi_name='TSI_' + str(r) + '_' + str(s)
    TSI = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df


def accumulation_distribution(df, n, acc_dist_name, column_name = None):
    """Calculate Accumulation/Distribution for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    ad = (2 * df[column_name] - df[conf_high] - df[conf_low]) / (df[conf_high] - df[conf_low]) * df[conf_volume]
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    #AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
    #df = df.join(AD)
    AD = pd.Series(ROC)
    #acc_dist_name = 'Acc/Dist_ROC_' + str(n)
    df[acc_dist_name] = AD.values
    return df


def chaikin_oscillator(df, Chaikin_name, column_name = None):
    """Calculate Chaikin Oscillator for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    ad = (2 * df[column_name] - df[conf_high] - df[conf_low]) / (df[conf_high] - df[conf_low]) * df[conf_volume]
    # Chaikin_name = 'Chaikin'
    Chaikin = pd.Series(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(), name = Chaikin_name)
    df = df.join(Chaikin)
    return df


def money_flow_index(df, n, mfi_name, column_name = None):
    """Calculate Money Flow Index and Ratio for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    PP = (df[conf_high] + df[conf_low] + df[column_name]) / 3
    i = 0
    PosMF = [0]
    while i < (len(df.index)-1):
        PosMF1 = 0
        if PP[i + 1] > PP[i]:
            PosMF1 = PP[i + 1] * df.loc[df.index[i+1], conf_volume]
        PosMF1 = PosMF1 / (PP[i] * df.loc[df.index[i], conf_volume])
        PosMF.append(PosMF1)
        i = i + 1
    #TotMF = PP * df[conf_volume]
    #MFR = pd.Series(PosMF_ser / TotMF)
    MFI = pd.Series(pd.Series(PosMF).rolling(n, min_periods=n).mean())
    # mfi_name = 'MFI_' + str(n)
    df[mfi_name] = MFI.values
    return df


def on_balance_volume(df, n, obv_name, column_name = None):
    """Calculate On-Balance Volume for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    OBV = [0]
    while i < (len(df.index)-1):
        if df.loc[df.index[i+1], column_name] - df.loc[df.index[i], column_name] > 0:
            OBV.append(df.loc[df.index[i+1], conf_volume])
        elif df.loc[df.index[i+1], column_name] - df.loc[df.index[i], column_name] < 0:
            OBV.append( 0- (df.loc[df.index[i+1], conf_volume]))
        else:
            OBV.append(0)
        i = i + 1

    OBV_ma = pd.Series(pd.Series(OBV).rolling(n, min_periods=n).mean())
    # obv_name = 'OBV_' + str(n)
    df[obv_name] = OBV_ma.values
    return df


def force_index(df, n, force_name, column_name = None):
    """Calculate Force Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    #force_name='Force_' + str(n)
    F = pd.Series(df[column_name].diff(n) * df[conf_volume].diff(n), name = force_name)
    df = df.join(F)
    return df


def ease_of_movement(df, n, eom_name, column_name = None):
    """Calculate Ease of Movement for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    EoM = (df[conf_high] - df[conf_low]) * ((df[conf_high] + df[conf_low]) - (df[conf_high].diff(1) + df[conf_low].diff(1))) / (2 * df[conf_volume])
    #eom_name = 'EoM_' + str(n)
    Eom_ma = pd.Series(EoM.rolling(n, min_periods=n).mean(), name = eom_name)
    df = df.join(Eom_ma)
    return df


def commodity_channel_index(df, n, cci_name, column_name = None):
    """Calculate Commodity Channel Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    ratio = 0.015
    PP = (df[conf_high] + df[conf_low] + df[column_name]) / 3
    #cci_name ='CCI_' + str(n)
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                    name=cci_name).divide(ratio)
    df = df.join(CCI)
    return df


def coppock_curve(df, n, Copp_name, column_name = None):
    """Calculate Coppock Curve for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    M = df[column_name].diff(int(n * 11 / 10) - 1)
    N = df[column_name].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df[column_name].diff(int(n * 14 / 10) - 1)
    N = df[column_name].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    # Copp_name = 'Copp_' + str(n)
    Copp = pd.Series((ROC1 + ROC2).ewm(span=n, min_periods=n).mean(), name=Copp_name)
    df = df.join(Copp)
    return df


def keltner_channel(df, n, column_name = None):
    """Calculate Keltner Channel for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    KelChM = pd.Series(((df[conf_high] + df[conf_low] + df[column_name]) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df[conf_high] - 2 * df[conf_low] + df[column_name]) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df[conf_high] + 4 * df[conf_low] + df[column_name]) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df


def ultimate_oscillator(df, Ultimate_Osc_name, column_name = None):
    """Calculate Ultimate Oscillator for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < (len(df.index)-1):
        TR = max(df.loc[df.index[i + 1], conf_high], df.loc[df.index[i], column_name])
        TR = TR - min(df.loc[df.index[i + 1], conf_low], df.loc[df.index[i], column_name])
        TR_l.append(TR)
        BP = df.loc[df.index[i + 1], column_name]
        BP = BP - min(df.loc[df.index[i + 1], conf_low], df.loc[df.index[i], column_name])
        BP_l.append(BP)
        i = i + 1

    rates = 4+2+1
    UltO = pd.Series(((4 * pd.Series(BP_l).rolling(7).sum() / pd.Series(TR_l).rolling(7).sum()) + (
                      2 * pd.Series(BP_l).rolling(14).sum() / pd.Series(TR_l).rolling(14).sum()) + (
                          pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l).rolling(28).sum())) / rates)
    # Ultimate_Osc_name = 'Ultimate_Osc'
    df[Ultimate_Osc_name] = UltO.values
    return df


def donchian_channel(df, n, donchian_name, column_name = None):
    """Calculate donchian channel of given pandas data frame.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    i = 0
    dc_l = []
    while i < n :
        dc_l.append(0)
        i += 1

    i = 0
    while (i + n - 1) < (len(df.index)-1):
        dc = max(df[conf_high].ix[i:i + n - 1]) - min(df[conf_low].ix[i:i + n - 1])
        dc_l.append(dc)
        i += 1

    donchian_chan = pd.Series(dc_l)
    donchian_chan = donchian_chan.shift(n - 1)
    # donchian_name = Donchian_' + str(n)
    df[donchian_name] = donchian_chan.values
    return df



def standard_deviation(df, n, std_name, column_name = None):
    """Calculate Standard Deviation for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    if column_name is None:
        column_name = conf_close

    # std_name = 'STD_' + str(n)
    df = df.join(pd.Series(df[column_name].rolling(n, min_periods=n).std(), name=std_name))
    df.fillna(0, inplace=True )  #fill nan with 0
    return df

