import json

all_conf = []
with open('qalgo_config.json') as f:
  all_conf = json.load(f)

conf_sp500_url = all_conf["sp500_url"]
conf_data_path = all_conf["data_path"]
conf_rawdata_path = all_conf["rawdata_path"]
conf_backtest_data_path = all_conf["backtest_data_path"]
conf_sp500_name = all_conf["sp500_name"]
conf_date = all_conf["ohlv"][0]
conf_open = all_conf["ohlv"][1]
conf_high = all_conf["ohlv"][2]
conf_low  = all_conf["ohlv"][3]
conf_close = all_conf["ohlv"][4]
conf_adjclose = all_conf["ohlv"][5]
conf_volume = all_conf["ohlv"][6]

