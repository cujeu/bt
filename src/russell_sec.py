#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
#import datetime
import argparse
from config import *
from sp500_symbols import *

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(description='ATR Startegy')

    # Defaults for dates
    parser.add_argument('--sector', '-s', required=False, default='none',
                        help='russell sectors')
    parser.add_argument('--industry', '-i', required=False, default='none',
                        help='russell industry')
    parser.add_argument('--ticker', '-t', required=False, default='none',
                        help='russell ticker')
    parser.add_argument('--list', '-l', required=False, default='none',
                        help='russell fundamental')
    parser.add_argument('--pretty', '-p', required=False, default='none',
                        help='clean % in dataframe')

    return parser.parse_args(pargs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python russell_sec.py -s sec_name -i ind_name -t ticker -l 'AAPL,GOOG'")
        slist = ['Computers and Technology', 'Retail-Wholesale', 'Auto-Tires-Trucks', 'Finance', 'Business Services', 'Medical', 'Consumer Discretionary', 'Consumer Staples', 'Oils-Energy', 'Utilities', 'Aerospace', 'Conglomerates', 'Transportation', 'Industrial Products', 'Construction', 'Basic Materials']
        for s in slist:
            print(s)
        sys.exit()
    args = parse_args()

    #for a, d in ((getattr(args, x), x) for x in ['sector', 'industry', 'ticker']):
    #    if d == 'sector':
    if args.sector != 'none':
        slist = get_russell_sectors()
        for sec in slist:
            ilist = get_russell_industry_sector(sec)
            print("\n====== "+sec+" =======")
            print(ilist)
        print("\n====== Russell Sectors =========")
        print(slist)
    elif args.industry != 'none':
        print("== Industry " +args.industry)
        print(get_russell_symbols_by_ind(args.industry))
    elif args.ticker != 'none':
        t = args.ticker.upper()
        print("== Ticker " + t)
        print(get_info_by_sym(t))
    elif args.list != 'none':
        slist = args.list.split(',')
        print('Sector','Industry','Market Cap', 'Net Income(a)',
              '5Y Rev%','ROE%','Debt/Equity','Price/Cash Flow')
        for s in slist:
            print(s + ':' + str(get_info_by_sym(s)))
    elif args.pretty != 'none':
        refine_russell_data()
        print("data washed")

