import pandas as pd 
import os 
import yaml
from dotenv import load_dotenv
load_dotenv()
from src.utilities.data_modelling_utilities import *
from src.logger import logging 

config_path = os.path.join('main_config.yml')
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_modelling_config_path']), 'r') as f:
    config = yaml.safe_load(f)

class Experiment_tracking_and_modelling():
    def __init__(self,train_data,test_data, dataframe_name, version):
        self.final_dataframe = pd.DataFrame(columns=["precision","recall","f1-score","support","model_name","dataframe_name","accuracy"])
        self.final_dataframe_K_scores = pd.DataFrame()
        self.train_data = train_data
        self.test_data = test_data
        self.dataframe_name = dataframe_name
        self.version = version
        self.result_dataframe = pd.DataFrame()
        self.k_scores_dataframe = pd.DataFrame()
    def initiate_modelling(self):
        target_column = config['target_column']
        self.x_train = self.train_data.drop(columns=[target_column], axis=1).reset_index(drop=True)
        self.y_train = self.train_data[[target_column]].reset_index(drop=True)
        self.x_test = self.test_data.drop(columns=[target_column], axis=1).reset_index(drop=True)
        self.y_test = self.test_data[[target_column]].reset_index(drop=True)

        logging.info("initiate modelling finished successfully")
        
    def run_experiment_tracking(self):
        self.result_dataframe,self.k_scores_dataframe,self.sc = modelling_function_experiment_tracking(self.x_train,
                                                                                               self.x_test,
                                                                                               self.y_train,
                                                                                               self.y_test,
                                                                                               'dataframe_1',
                                                                                                config['no_of_folds'],
                                                                                                config['target_column'],
                                                                                                self.version)
        self.k_scores_dataframe = self.k_scores_dataframe.reset_index(names = 'Iteration')
        self.result_dataframe = self.result_dataframe.reset_index(names = 'class')
        logging.info("experient tracking finished successfully")