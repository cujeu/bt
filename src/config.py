import json

all_conf = []
with open('qalgo_config.json') as f:
  all_conf = json.load(f)

conf_sp500_url = all_conf["sp500_url"]
conf_app_path = all_conf["app_path"]
conf_data_path = all_conf["data_path"]
conf_rawdata_path = all_conf["rawdata_path"]
conf_backtest_data_path = all_conf["backtest_data_path"]
conf_data_path = conf_app_path + conf_data_path
conf_rawdata_path = conf_app_path + conf_rawdata_path
conf_backtest_data_path = conf_app_path + conf_backtest_data_path

conf_sp500_name = all_conf["sp500_name"]
russell1000_name = all_conf["russell1000_name"]
conf_date = all_conf["ohlv"][0]
conf_open = all_conf["ohlv"][1]
conf_high = all_conf["ohlv"][2]
conf_low  = all_conf["ohlv"][3]
conf_close = all_conf["ohlv"][4]
conf_adjclose = all_conf["ohlv"][5]
conf_volume = all_conf["ohlv"][6]

#url of stock symbols of ark etf
arklinks = ["https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv",
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv"
           ]

# dataframe column 
conf_mk_view_col = ["sym", "chg5", "chg10", "csU0", "cs", "cs-1", "sm", "ml","ema20","ema20-2","ema60"]
conf_ticker_idx =   0
conf_chg5_idx =     1
conf_chg10_idx =    2
conf_cs0Cnt_idx =   3
conf_cs_idx =       4
conf_csChg_idx =    5
conf_sm_idx =       6
conf_ml_idx =       7
conf_ema20_idx =    8
conf_ema20Chg_idx = 9
conf_ema60_idx =    10

