# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
from datetime import timedelta
from _utils.customlogger import customlogger as CL
from _utils.visualization_plot import *

pd.set_option('display.max_colwidth', -1)    #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)        # display 50개 까지 

# In[ ]:
# ** loading config **
with open('./../{}'.format("cohort.json")) as file:
    cfg = json.load(file)
# In[ ]:
# ** loading info **
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent
current_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(__file__))[0]

# In[ ]:
# **create Logger**
log = CL("custom_logger")
pathlib.Path.mkdir(pathlib.Path('{}/_log/'.format(parent_dir)), mode=0o777, parents=True, exist_ok=True)
log = log.create_logger(file_name="../_log/{}.log".format(curr_file_name), mode="a", level="DEBUG")    
log.debug('start {}'.format(curr_file_name))

# In[ ]:
# ** connection DataBase **
if (cfg["dbms"]=="postgresql"):
    db_cfg = cfg["postgresql"]
    import psycopg2 as pg
    conn = pg.connect(host=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], dbname=db_cfg['@database']) 
    log.debug("postgresql connect")
    
elif (cfg["dbms"]=="mssql"):
    db_cfg = cfg["mssql"]
    import pymssql
    conn= pymssql.connect(server=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], database=db_cfg['@database'], as_dict=False)
    log.debug("mssql connect")
    
else:
    log.warning("set config.json - sql - dbms : mssql or postgresql")

# In[ ]:
def renderTranslateQuerySql(sql_query, dict):
    '''
    ----- Replace text ------
    "@cdm_database_schema",
    "@target_database_schema",
    "@target_cohort_table",
    "@vocabulary_database_schema",
    "@target_cohort_id"
    ----- Replace text ------
    '''
    for key, value in dict.items():
        sql_query = sql_query.replace(key, value)
    return sql_query

try :
    pass
except :
    traceback.print_exc()
    log.error(traceback.format_exc())

# In[ ]:
for drug in cfg['drug'].keys():
    print(drug)

# In[ ]:
for atccode in cfg['atccode'].keys():
    
    try :
        output_dir = pathlib.Path('{}/result/{}/data_distribution/{}'.format(parent_dir, current_date, atccode))
        pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)
        print(atccode)
        drug_set = cfg['atccode'][atccode]['drugset']
        drug_df_list = []
        for drug in drug_set:
            print(drug)
            # sql_query = "select * from {}.person_{}".format(db_cfg['@person_database_schema'], drug)
            sql_query = "select * from {}.person_{}".format('temp_suncheol.dbo', drug)
            print(sql_query)
            person_df = pd.read_sql(sql=sql_query, con=conn)
            drug_df_list.append(person_df)
            print(len(person_df))
        concat_df = pd.concat(drug_df_list, axis=0)
        print(len(concat_df))
        concat_df.drop_duplicates(subset=['person_id'], keep='first', inplace=True)

        saveGenderPiePlot(concat_df, output_dir, drug+"_gender1")
        saveGenderbarPlot(concat_df, output_dir, drug+"_gender2")
        saveAgebarPlot(concat_df, output_dir, drug+"_age")
            
    except :
        traceback.print_exc()
        log.error(traceback.format_exc())
                

# In[ ]:
from matplotlib import pyplot as plt
import seaborn as sns

    
with conn.cursor() as cursor:
    for drug in cfg['drug'].keys():
        try :
            output_dir = pathlib.Path('{}/result/{}/create_cohort/{}'.format(parent_dir, current_date, drug))
            pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)
            
            sql_query = "select * from {}.person_{}".format(db_cfg['@person_database_schema'], drug)
            population_df = pd.read_sql(sql=sql_query, con=conn)
            population_df.drop_duplicates(subset=['person_id'], keep='first', inplace=True)

            saveGenderPiePlot(population_df, output_dir, drug+"_gender1")
            saveGenderbarPlot(population_df, output_dir, drug+"_gender2")
            saveAgebarPlot(population_df, output_dir, drug+"_age")
                    
        except :
            traceback.print_exc()
            log.error(traceback.format_exc())
        
    conn.commit()

# In[ ]:
conn.close()
# %%
