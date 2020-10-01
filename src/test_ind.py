import pandas as pd
import numpy as np
from config import *
from ind import *

if __name__ == "__main__":
    tick = 'AAPL'
    file_name = conf_rawdata_path + tick + '.csv'
    data = pd.read_csv(file_name) 
    # Preview the first 5 lines of the loaded data 
    ma_name='MA_' + str(10)
    mv_df = moving_average(data, 10, ma_name)
    print(mv_df.head())
    print(mv_df.tail())

    """
    ## test std
    data = pd.read_csv(file_name) 
    # Preview the first 5 lines of the loaded data 
    ma_name='STD_' + str(10)
    mv_df = standard_deviation(data, 10, ma_name)
    print(mv_df.head())
    print(mv_df.tail())
    """
    ## test std
    data = pd.read_csv(file_name) 
    # Preview the first 5 lines of the loaded data 
    ma_name='STC_' + str(10),
    mv_df = stc(mv_df)
    print(mv_df.head())
    print(mv_df.tail())

