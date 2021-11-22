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
                        help='russell infomation')
    parser.add_argument('--pool', '-p', required=False, default='none',
                        help='growth or strong russell pool')
    parser.add_argument('--wash', '-w', required=False, default='none',
                        help='clean dataframe')

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
    if args.sector == 'all':
        slist = get_russell_sectors()
        for sec in slist:
            ilist = get_russell_industry_sector(sec)
            print("\n====== "+sec+" =======")
            print(ilist)
        print("\n====== Russell Sectors =========")
        print(slist)
    elif args.sector != 'none':
        sec_name = args.sector
        sec_list = ['Computers and Technology', 'Basic Materials', 'Transportation', 'Retail-Wholesale', 'Medical', 'Finance', 'Consumer Staples', 'Industrial Products', 'Business Services', 'Utilities', 'Auto-Tires-Trucks', 'Oils-Energy', 'Consumer Discretionary', 'Construction', 'Aerospace', 'Conglomerates']
        if sec_name in sec_list:
            sort_russell_data(True, sec_name)
        else:
            sort_russell_data(False, sec_name)

    elif args.industry == 'all':
        print("== all Industry ")
        sort_russell_by_ind()
    elif args.industry != 'none':
        print("== Industry " +args.industry)
        print(get_russell_symbols_by_ind(args.industry))
    elif args.ticker != 'none':
        t = args.ticker.upper()
        print("== Ticker " + t)
        print(get_info_by_sym(t.upper()))
    elif args.list != 'none':
        slist = args.list.split(',')
        print('Sector','Industry','Market Cap', 'Net Income(a)',
              '5Y Rev%','ROE%','Debt/Equity','Price/Cash Flow','P/E ttm')
        for s in slist:
            print(s + ':' + str(get_info_by_sym(s.upper())))
    elif args.wash != 'none':
        refine_russell_data()
        print("data washed")
    elif args.pool == 'strong':
        print(" Sales>100M, Net Income>100M, 5Y Rev grownth>50, Debt/Equity<2")
        tlist = get_strong_russell_symbols()
        print(tlist)
        print('strong')
    elif args.pool == 'growth':
        print(" Sales>100M, Net Income>100M, 5Y Rev grownth>50, Debt/Equity<2")
        tlist = get_growth_russell_symbols()
        print(tlist)
        print('growth')

