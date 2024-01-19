import pandas as pd
import pandasql 
from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text as sql_text
from datetime import date 

from dotenv import load_dotenv
load_dotenv()
import os 
connection_url = URL.create(drivername='mysql',
                            username=os.getenv("DB_USERNAME"),
                            password=os.getenv("DB_PASSWORD"),
                            host=os.getenv("DB_HOST"),
                            database=os.getenv("DB_NAME"),
                            query={"ssl_ca": os.getenv("CA_CERT_PATH")})
engine = create_engine(connection_url)

def dataframe_to_sql( dataframe , table_name, version, new_table = 'NO'):
    if new_table == 'NO':
        dataframe['version'] = version
        dataframe['datetime_stamp'] = str(date.today())
        dataframe.to_sql(table_name, con = engine, if_exists = 'append', index = False)  
    else:
        dataframe['version'] = version
        dataframe['datetime_stamp'] = str(date.today())
        dataframe.to_sql(table_name, con = engine, if_exists = 'replace',index = False)  
