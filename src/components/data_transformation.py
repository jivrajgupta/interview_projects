import pandas as pd 
import os 
import yaml
from dotenv import load_dotenv
load_dotenv()
from src.utilities.data_transformation_utilities import *
from src.logger import logging 

config_path = os.path.join('main_config.yml')
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_transformation_config_path']), 'r') as f:
    config = yaml.safe_load(f)


class initiate_data_transformation():
    def __init__(self,data):
        self.data = data 
    
    def data_transformation(self):
        for column in config['original_column_list']:
            remove_null_value(self.data,column)
        for column in config['categorical_columns']:
            self.data = feature_addition_frequency_encoding(self.data,column,config['target_column'])
        for column in config['datetime_columns']:
            self.data = self.data.drop(column, axis = 1)
        return self.data
    
    
    
    