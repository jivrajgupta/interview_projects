import pandas as pd 
import os 
import yaml
from dotenv import load_dotenv
load_dotenv()
from src.logger import logging 

config_path = os.path.join('main_config.yml')
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_transformation_config_path']), 'r') as f:
    config = yaml.safe_load(f)

def remove_null_value(dataframe, columns_to_remove):
    dataframe.dropna(subset=columns_to_remove, inplace=True)
    return dataframe

def transformation_func(df,col_list_1,col_list_2,cutoff_percentage): 
    if col_list_1 is not None:
        temp_df = pd.DataFrame(df.groupby(col_list_1)[col_list_2].value_counts(normalize=True) < cutoff_percentage)
        temp_df = temp_df[temp_df[col_list_2] == True]
        excluded_list = temp_df.index.get_level_values(col_list_2).tolist()
        df[col_list_2] = df[col_list_2].map(lambda x: 'Others' if x in excluded_list else x)
        return df[col_list_2]
    else:
        temp_df = pd.DataFrame(df[col_list_2].value_counts(normalize=True) < cutoff_percentage)
        temp_df = temp_df[temp_df[col_list_2] == True]
        excluded_list = temp_df.index.tolist()
        df[col_list_2] = df[col_list_2].map(lambda x: 'Others' if x in excluded_list else x)
        return df[col_list_2]

def feature_addition_target_encoding(temp_df, col, target_column):
    dict = temp_df.groupby([col])[target_column].mean().to_dict()
    new_column_name = col + '_' + "target_encoded_" + target_column
    temp_df[new_column_name] = temp_df[col].map(dict)
    return temp_df.drop(columns=[col])
def feature_addition_frequency_encoding(temp_df,col,target_column):
    dict = temp_df[col].value_counts(normalize=True).to_dict()
    new_column_name = col + '_' + 'count_encoded'
    temp_df[new_column_name] = temp_df[col].map(dict)
    return temp_df.drop(columns=[col])
def feature_additon_OHE_encoding(temp_df,col):
    dummy = pd.get_dummies(data=temp_df[col], prefix='OHE', drop_first=True)
    OHE_group_name = dummy.columns.tolist()
    temp_df_1 = pd.merge(temp_df.drop(columns=[col]), dummy, left_index=True, right_index=True)
    return temp_df_1