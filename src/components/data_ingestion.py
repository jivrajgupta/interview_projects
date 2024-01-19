import os
import sys
from utilities import dataframe_to_sql
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import yaml
from src.utilities.data_ingestion_utilities import *
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sqlalchemy import URL, inspect
from dotenv import load_dotenv
from utilities import * 
from sqlalchemy import create_engine, text as sql_text
load_dotenv()
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_ingestion_config_path']), 'r') as f:
    config = yaml.safe_load(f)


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train_dataset.csv')
    test_data_path:str=os.path.join('artifacts','test_dataset.csv')
    raw_data_path:str=os.path.join('artifacts','raw_dataset.csv')


class DataIngestion:
    def __init__(self,version):
        self.ingestion_config=DataIngestionConfig()
        self.connection_url = URL.create(
                                        drivername='mysql',
                                        username=os.getenv("DB_USERNAME"),
                                        password=os.getenv("DB_PASSWORD"),
                                        host=os.getenv("DB_HOST"),
                                        database=os.getenv("DB_NAME"),
                                        query={"ssl_ca": os.path.join("cacert.pem")} 
                                        )
        self.version = version 
    def initiate_data_ingestion(self):
        try:
            self.engine = create_engine(self.connection_url)
            logging.info("Connection has been established to the SQL database")
            os.makedirs("artifacts",exist_ok=True)
            if os.path.exists("artifacts/raw_dataset.csv"):
                self.df = pd.read_csv(self.ingestion_config.raw_data_path)
            else:
                query = 'select * from raw_dataset limit 10000'
                self.df = pd.read_sql_query(sql=sql_text(query),con = self.engine.connect())
                self.df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Reading completed mysql database")
            logging.info("Raw data file has been extracted")


            train_set,test_set=train_test_split(self.df,test_size=config['test_size'],random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("train_test split has been performed and files have been saved")

            self.df_info = get_df_info(self.df,self.version)      
            self.category_count = category_count_df(self.df,self.version)
            self.dataset_comments = dataset_comments(self.df,self.version)

            # self.column_validation_check = column_validation_check(self.df)
            # self.column_validation_check.to_sql('column_validation_check',con = self.engine,index = False,if_exists = 'replace')
            # column_validation_check.to_csv(os.path.join(versioned_folder_dataset_info_path,r"column_validation_check.csv"))
            self.category_count.to_csv(os.path.join("versioned_folder_dataset_info_path",r"category_count.csv"),header=True)
            self.df_info.to_csv(os.path.join("versioned_folder_dataset_info_path",r"df_info.csv"),index=False,header=True)
            self.dataset_comments.to_csv(os.path.join("versioned_folder_dataset_info_path",r"dataset_comments.csv"),index=False,header=True)

            logging.info("***************Data Ingestion is completed***********************")

        except Exception as e:
            logging.error(CustomException(e,sys))