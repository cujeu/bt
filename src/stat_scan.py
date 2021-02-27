# price divergence factor for ETF poll
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time 
import datetime  
import os
import argparse
#import sys  
import numpy as np
#import random
#import pickle 
import matplotlib.pyplot as plt
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
from config import *
from sp500_symbols import *
from inst_ind import *

def run_stat(target_csv):
    filename = conf_data_path + 'scan_'+ target_csv + '.csv'
    #mk_df = pd.DataFrame(columns = ["sym", "date", "close", "high", "perc", "exit","edate","eperc"])
    #mk_df= pd.read_csv(filename) #, index_col=0, parse_dates=True)
    mk_df= pd.read_csv(filename, index_col=[0]) #, index_col=0, parse_dates=True)

    #view #1: use hist only
    #mk_df.hist(column=['perc','eperc'])

    #view #2: use bins
    #mk_df.plot.hist(bins=10, column='perc') #alpha=0.5)

    step = 5
    #bin_range = np.arange(0, 100+step, step)
    bin_range = np.arange(-2, 120, step)
    out, bins  = pd.cut(mk_df['perc'], bins=bin_range, include_lowest=True, right=False, retbins=True)
    out.value_counts(sort=False).plot.bar()
    # verify in libreoffice: =COUNTIF(F1:F411, "<"&L2) - COUNTIF(F1:F411, "<"&K2)
    # for example: L2=30 K2=20 to count percentage between 20% and 30%
    plt.show()
    #print(mk_df)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(description='Data Stat')

    # Defaults for dates
    parser.add_argument('--target', '-t', required=False, default='etf',
                        help='etf/ark/rusell/sp500')

    parser.add_argument('--plot', '-p', required=False, default='yes',
                        help='plot or not')


    return parser.parse_args(pargs)

if __name__ == '__main__':
    
    args = parse_args()

    # Parse ark etf
    for a, d in ((getattr(args, x), x) for x in ['target', 'plot']):
        if d == 'target':
            run_stat(a)
        elif d == 'plot':
            print(a)

