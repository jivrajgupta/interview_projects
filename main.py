from src.logger import logging 
from src.components.data_transformation import initiate_data_transformation
from src.components.data_modelling import Experiment_tracking_and_modelling
import pandas as pd
import os 
import mlflow
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from sqlalchemy import inspect
from utilities import * 
import sys
from dotenv import load_dotenv
load_dotenv()
import os 
connection_url = URL.create(drivername='mysql',
                            username=os.getenv("DB_USERNAME"),
                            password=os.getenv("DB_PASSWORD"),
                            host=os.getenv("DB_HOST"),
                            database=os.getenv("DB_NAME"),
                            query={"ssl_ca":"cacert.pem"})
engine = create_engine(connection_url)
inspector = inspect(engine)
if 'version' in inspector.get_table_names():
    query = f'select max(datetime_stamp) as maximum_date, max(version_id) as maximum_version from version'
    verison_df = pd.read_sql_query(sql = sql_text(query),con = engine.connect())
    if str(verison_df['maximum_date'][0]) == None:
        version = 1
    elif str(verison_df['maximum_date'][0]) == str(date.today()):
        version = verison_df['maximum_version'][0] + 1
    else:
        version = 1
else:
    version = 1
    sql_command = f"create table version (datetime_stamp date, version_id int)"
    con = engine.connect()
    con.execute(sql_text(sql_command))
    con.commit()
    con.close()
    

try:
    logging.info("main file has started")

    data_ingestion = DataIngestion(version)
    data_ingestion.initiate_data_ingestion()

    logging.info("initiate_data_transformation")

    train_data = pd.read_csv(os.path.join("artifacts/train_dataset.csv"))
    test_data = pd.read_csv(os.path.join("artifacts/test_dataset.csv"))

    train_data_treated = initiate_data_transformation(train_data).data_transformation()
    test_data_treated = initiate_data_transformation(test_data).data_transformation()

    train_data_treated.to_csv("train_treated.csv")
    test_data_treated.to_csv("test_treated.csv")


    logging.info("********************data_transformation_finished*************************")

    logging.info("initiating model function")

    mlflow_model = Experiment_tracking_and_modelling(train_data_treated,test_data_treated,'dataframe_testing',version)
    mlflow_model.initiate_modelling()
    mlflow_model.run_experiment_tracking()
   
    scaler_train = mlflow_model.sc
    logging.info("main file has finished")


    # reg_model = mlflow.search_model_versions()
    # prod = [model for model in reg_model if model.current_stage == 'Production']
    # model = prod[0]
    # model_prod_1 = mlflow.pyfunc.load_model(model_uri = f"models:/{model.name}/{model.version}")
    # test_data_treated = scaler_train.transform(test_data_treated.drop(columns = 'is_canceled'))
    # test_data_treated['predicted'] = model_prod_1.predict(test_data_treated.drop(columns = 'is_canceled'))
    # test_data_treated.to_csv("predictions.csv")
except Exception as e:
    logging.error(CustomException(e,sys))

logging.info("logging all results to sql")

sql_command = f"INSERT INTO version VALUES ('{str(date.today())}','{version}')"
con = engine.connect()
con.execute(sql_text(sql_command))
con.commit()
con.close()

try:
    if 'dataset_info' in inspector.get_table_names():
        dataframe_to_sql(data_ingestion.df_info,'dataset_info',version)      
    else:
        dataframe_to_sql(data_ingestion.df_info,'dataset_info',version,'Yes')      
    if 'dataset_comments' in inspector.get_table_names():
        dataframe_to_sql(data_ingestion.dataset_comments.reset_index(drop = True),'dataset_comments',version)
    else:
        dataframe_to_sql(data_ingestion.dataset_comments.reset_index(drop = True),'dataset_comments',version,'Yes')
    if 'category_counts' in inspector.get_table_names():
        dataframe_to_sql(data_ingestion.category_count.reset_index(),'category_counts',version) 
    else:
        dataframe_to_sql(data_ingestion.category_count.reset_index(),'category_counts',version,'Yes')

    if 'model_results' in inspector.get_table_names():
        dataframe_to_sql(mlflow_model.result_dataframe,'model_results',version) 
    else:
        dataframe_to_sql(mlflow_model.result_dataframe,'model_results',version,'Yes') 
        
    if 'K_scores_train_test' in inspector.get_table_names():
        dataframe_to_sql(mlflow_model.k_scores_dataframe,'K_scores_train_test',version) 
    else:
        dataframe_to_sql(mlflow_model.k_scores_dataframe,'K_scores_train_test',version,'Yes')  

    logging.info("all tables logged to sql")
except Exception as e:
    logging.error(e,sys)  
