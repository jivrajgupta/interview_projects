import pandas as pd
import os 
from datetime import datetime
from src.logger import logging
import numpy as np
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
import itertools
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
import re
from datetime import datetime, date
config_path = os.path.join('main_config.yml')
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_ingestion_config_path']), 'r') as f:
    config = yaml.safe_load(f)

def dataset_comments(df,version):
    df_print = pd.DataFrame(columns = ['Comments','Value'])
    df_print.loc[len(df_print)] = {'Comments':'orignal_Columns_Under_Consideration','Value':len(df.columns.to_list())}
    df_print.loc[len(df_print)] = {'Comments':'Categorical_Columns_Under_Consideration',
                                'Value':len(config['categorical_columns'])}
    df_print.loc[len(df_print)] = {'Comments':'Numerical_Columns_Under_Consideration',
                                'Value':len(config['numerical_columns'])}
    df_print.loc[len(df_print)] = {'Comments':'DaTetime_Columns_Under_Consideration',
                                'Value':len(config['datetime_columns'])}
    df_print.loc[len(df_print)] = {'Comments':'Target_Column_Under_Consideration','Value':str(config['target_column'][0])}
    df_print['datetime_stamp']  = date.today()
    df_print['version'] = version 
    df_print['new_column_1'] = 'jivraj1'
    df_print['new_column_2'] = 'jivraj_2'
    return df_print
    

def get_df_info(dataframe,version, include_unique=True):
    column = [col for col in dataframe.columns]
    column_type = [type(cell) for cell in dataframe.loc[0, :]]
    null_count = [dataframe[col].isna().sum() for col in dataframe.columns]
    null_percent = [((dataframe[col].isna().sum() / dataframe.shape[0]) * 100) for col in dataframe.columns]
    
    if include_unique:
        unique_count = [dataframe[col].nunique() for col in dataframe.columns]
        dataframe_info = pd.DataFrame({'column': column, 'column_type': column_type,
                                       'null_count': null_count, 'unique_count': unique_count,
                                       'null_percent': null_percent})
    else:
        dataframe_info = pd.DataFrame({'column': column, 'column_type': column_type,
                                       'null_count': null_count, 'null_percent': null_percent})
    dataframe_info['datetime_stamp'] = date.today()
    dataframe_info['version'] = version
    return dataframe_info.sort_values(by='null_count', ascending=False)



def category_count_df(df,version):
    category_count = pd.DataFrame()
    config['categorical_columns']
    for column in df[config['categorical_columns']]:
        value_counts = df[column].value_counts().sort_index().sort_values(ascending = False)
        percentages = (value_counts / len(df[column])) * 100
        column_data = pd.concat([value_counts, percentages], axis=1, keys=['Count', 'Percentage'])
        column_data.index = pd.MultiIndex.from_product([[column], column_data.index], names=['Column', 'Category'])
        category_count = pd.concat([category_count, column_data]) 
    category_count['datetime_stamp'] = date.today()
    category_count['version'] = version
    return category_count
            
            
def column_validation_check(df):
    set1 = set(config['numerical_columns'])
    set2 = set(config['categorical_columns'])
    set3 = set(config['datetime_columns'])
    set4 = set(config['original_column_list'])
    if not (set1 & set2) and not (set1 & set3) and not (set2 & set3) and set4 == (set1 | set2 | set3) :
        return 'Each list contains different columns and all columns have been captured'
    else:
        return 'More than one list contains the same column or there is a mismatch with total columns'