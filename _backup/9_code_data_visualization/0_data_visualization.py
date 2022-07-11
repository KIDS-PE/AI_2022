#!/usr/bin/env python
# coding: utf-8

'''
--------------
data visualization
--------------
'''

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
import seaborn as sns
import traceback
from math import sqrt
from tqdm import tqdm
from datetime import timedelta
from _utils.visualization_plot import *
from _utils.customlogger import customlogger as CL

# In[ ]:
# ** loading config **
with open('./../{}'.format("config.json")) as file:
    cfg = json.load(file)

# In[ ]:
# ** loading info **
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent
current_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(__file__))[0]

# In[ ]:
# ** create Logger **
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

for outcome_name in tqdm(cfg['meas'].keys()) :
    try :
        log.debug('meas : {}'.format(outcome_name))
        output_data_dir = pathlib.Path('{}/data/{}/visualization/'.format(parent_dir, current_date))
        pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
        output_result_dir = pathlib.Path('{}/result/{}/visualization/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

        # In[ ]:
        # - Table Load from DB and Save dataset
        tnPopulation = '{}.person_{}'.format(db_cfg['@person_database_schema'], outcome_name)
        sql_query = 'select * from {}'.format(tnPopulation)
        population_df = pd.read_sql(sql=sql_query, con=conn)
        population_df.to_csv('{}/person_{}.txt'.format(output_data_dir, outcome_name),index=False)

        # In[ ]:
        # - load dataset
        population_df=pd.read_csv('{}/person_{}.txt'.format(output_data_dir, outcome_name))
        population_df['age_dec'] = population_df.age.map(lambda age: 10 * (age // 10))
        outlier = population_df.query('age>120 or age <=0')
        population_df.drop(outlier.index, inplace=True)
        print(len(outlier), len(population_df))
        #population_df.plot(kind='scatter',x='age',y='value_as_number')

        '''
        Drawing plot (total set ; No Remove Outlier)
        '''
        save_JointPlot(df=population_df, filedir=output_result_dir, filename=outcome_name)
        save_quadplot(df=population_df, filedir=output_result_dir, filename=outcome_name)
        save_percentile_plot(df=population_df, filedir=output_result_dir, filename=outcome_name)

        '''
        Drawing plot (remove outlier ; 3-IQR rule)
        '''
        q1=population_df['value_as_number'].quantile(0.25)
        q3=population_df['value_as_number'].quantile(0.75)
        iqr=q3-q1
        outlier = population_df[population_df['value_as_number']>q3+3*iqr].index
        population_df.drop(outlier, inplace=True)
        print(len(outlier))
        outlier = population_df[population_df['value_as_number']<q1-3*iqr].index
        population_df.drop(outlier, inplace=True)
        print(len(outlier))

        save_JointPlot(df=population_df, filedir=output_result_dir, filename=outcome_name+"(3-IQR)")
        save_quadplot(df=population_df, filedir=output_result_dir, filename=outcome_name+"(3-IQR)")
        save_percentile_plot(df=population_df, filedir=output_result_dir, filename=outcome_name+"(3-IQR)")

        output={}
        output['nPatient_male'] = len(population_df[population_df["gender_source_value"] == 'M'])
        output['nPatient_female'] = len(population_df[population_df["gender_source_value"] == 'F'])

        print(output['nPatient_male'], output['nPatient_female'])
        out = open('{}/output.txt'.format(output_result_dir),'a')

        out.write(str(outcome_name) + '///' )
        out.write(str(output['nPatient_male']) + '///')
        out.write(str(output['nPatient_female']) + '\n')
        out.close()

    except :
        traceback.print_exc()
        log.error(traceback.format_exc())

conn.close()

# %%
