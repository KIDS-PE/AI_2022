#!/usr/bin/env python
# coding: utf-8

'''
import SQL
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
import traceback
from tqdm import tqdm
from datetime import timedelta
from _utils.customlogger import customlogger as CL

pd.set_option('display.max_colwidth', -1)  #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)      # display 50개 까지 

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
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        log.debug('drug : {}'.format(outcome_name))
        
        # In[ ]:
        # ** create output(data / result) dir **
        output_dir = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)
        output_result_dir = pathlib.Path('{}/result/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

        # In[ ]:
        # ** set Tablename for reading from DB **
        tnPopulation = '{}.person_{}'.format(db_cfg["@person_database_schema"], outcome_name)
        tnMeasurement = '{}.measurement'.format(db_cfg["@cdm_database_schema"])
        tnDrug = '{}.drug_exposure'.format(db_cfg["@cdm_database_schema"])
        tnProcedure = '{}.procedure_occurrence'.format(db_cfg["@cdm_database_schema"])
        tnCondition = '{}.condition_occurrence'.format(db_cfg["@cdm_database_schema"])
        tnConcept = '{}.concept'.format(db_cfg["@cdm_database_schema"])

        # In[ ]:
        # ** read total population **
        sql_query = 'select * from {}'.format(tnPopulation)
        population_df = pd.read_sql(sql=sql_query, con=conn)
        log.debug('success : {}'.format(len(population_df)))

        # In[ ]:
        # ** Table Load from DB (Measurement / drug / procedure / concept) **
        sql_query ="select person_id, measurement_concept_id, measurement_date, value_as_number, range_low, range_high from {} ".format(tnMeasurement) +    "where measurement_concept_id!=0 and value_as_number is not null and person_id in (select distinct person_id from {})".format(tnPopulation)
        meas_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select person_id, drug_concept_id, drug_exposure_start_date, quantity from {} ".format(tnDrug) +    "where drug_concept_id!=0 and quantity is not null and person_id in (select distinct person_id from {})".format(tnPopulation)
        drug_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select person_id, procedure_concept_id, procedure_date from {} ".format(tnProcedure) +    "where procedure_concept_id!=0 and person_id in (select distinct person_id from {} )".format(tnPopulation)
        proc_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select person_id, condition_concept_id, condition_start_date from {} ".format(tnCondition) +    "where condition_concept_id!=0 and person_id in (select distinct person_id from {} )".format(tnPopulation)
        cond_df = pd.read_sql(sql=sql_query, con=conn)
        # check 
        log.debug('success : {}, {}, {}, {}'.format(len(meas_df), len(drug_df), len(proc_df), len(cond_df)))

        # In[ ]:
        # ** Table Load from DB (Concept) **
        sql_query="select distinct concept_id, concept_name from {} ".format(tnConcept) +    "where concept_id !=0 and concept_id in (select distinct measurement_concept_id from {})".format(tnMeasurement)
        concept_meas_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select distinct concept_id, concept_name from {} ".format(tnConcept) +    "where concept_id !=0 and concept_id in (select distinct drug_concept_id from {})".format(tnDrug)
        concept_drug_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select distinct concept_id, concept_name from {} ".format(tnConcept) +    "where concept_id !=0 and concept_id in (select distinct procedure_concept_id from {})".format(tnProcedure)
        concept_proc_df = pd.read_sql(sql=sql_query, con=conn)
        sql_query="select distinct concept_id, concept_name from {} ".format(tnConcept) +    "where concept_id !=0 and concept_id in (select distinct condition_concept_id from {})".format(tnCondition)
        concept_cond_df = pd.read_sql(sql=sql_query, con=conn)
        log.debug('success : {}, {}, {}, {}'.format(len(concept_meas_df), len(concept_drug_df), len(concept_proc_df), len(concept_cond_df)))
        concept_df = pd.concat([concept_meas_df, concept_drug_df, concept_proc_df, concept_cond_df], axis=0)

        # In[ ]:
        # ** Save dataset **

        population_df.to_csv('{}/population.txt'.format(output_dir),index=False)
        meas_df.to_csv('{}/measurement.txt'.format(output_dir),index=False)
        drug_df.to_csv('{}/drug.txt'.format(output_dir),index=False)
        proc_df.to_csv('{}/procedure.txt'.format(output_dir),index=False)
        cond_df.to_csv('{}/condition.txt'.format(output_dir),index=False)
        concept_df.to_csv('{}/concept.txt'.format(output_dir),index=False)

        # ### Load dataset

        # In[ ]:
        # ** Load dataset **

        # population_df=pd.read_csv('{}/population.txt'.format(output_dir))
        # meas_df=pd.read_csv('{}/measurement.txt'.format(output_dir))
        # drug_df=pd.read_csv('{}/drug.txt'.format(output_dir))
        # proc_df=pd.read_csv('{}/procedure.txt'.format(output_dir))
        # cond_df=pd.read_csv('{}/condition.txt'.format(output_dir))
        # concept_df=pd.read_csv('{}/concept.txt'.format(output_dir))

        # In[ ]:
        # population_df['label'] = (~population_df['first_abnormal_date'].isnull()).astype(int)
        population_df.rename(columns={'gender_source_value':'sex'}, inplace=True)
        population_df['sex'].replace(['F', 'Female'], 0, inplace=True)
        population_df['sex'].replace(['M', 'Male'], 1, inplace=True)

        # meas_df = meas_df[["person_id","measurement_concept_id","measurement_date","value_as_number"]]
        meas_df = meas_df[["person_id", "measurement_concept_id", "measurement_date", "value_as_number", "range_low", "range_high"]]
        drug_df = drug_df[["person_id","drug_concept_id","drug_exposure_start_date","quantity"]]
        proc_df = proc_df[["person_id","procedure_concept_id","procedure_date"]]
        cond_df = cond_df[["person_id","condition_concept_id","condition_start_date"]]
        concept_df = concept_df[["concept_id","concept_name"]]

        def drop_duplicates_(domain_df):
            n_prev = len(domain_df)
            domain_df = domain_df.drop_duplicates()
            n_next = len(domain_df)
            print('{}>{}'.format(n_prev, n_next))
            return domain_df

        meas_df = drop_duplicates_(meas_df)
        drug_df = drop_duplicates_(drug_df)
        proc_df = drop_duplicates_(proc_df)
        cond_df = drop_duplicates_(cond_df)

        ### @use common terminology.
        meas_df.rename(columns={'measurement_concept_id':'concept_id','measurement_date':'concept_date','value_as_number':'concept_value'}, inplace=True)
        drug_df.rename(columns={'drug_concept_id':'concept_id','drug_exposure_start_date':'concept_date','quantity':'concept_value'}, inplace=True)
        proc_df.rename(columns={'procedure_concept_id':'concept_id','procedure_date':'concept_date'}, inplace=True)
        cond_df.rename(columns={'condition_concept_id':'concept_id','condition_start_date':'concept_date'}, inplace=True)

        ### population + domain
        meas_df = pd.merge(population_df, meas_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
        drug_df = pd.merge(population_df, drug_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
        proc_df = pd.merge(population_df, proc_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
        cond_df = pd.merge(population_df, cond_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)

        # In[ ]:

        ### Get only used dates
        def cohortConditionSetting(domain_df, pre_observation_period, post_observation_peroid):
            from datetime import timedelta
            prev_len = len(domain_df)
            domain_df['cohort_start_date'] = pd.to_datetime(domain_df['cohort_start_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            # domain_df['first_abnormal_date'] = pd.to_datetime(domain_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            domain_df['concept_date'] = pd.to_datetime(domain_df['concept_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            # condition 1) Select patients with first adverse events within 2 months of cohort initiation.
            domain_df = domain_df[(domain_df['cohort_start_date']<=domain_df['concept_date']+timedelta(days=pre_observation_period))]
            # condition 2) Delete data before the cohort start date.
            domain_df = domain_df[(domain_df['concept_date']<=domain_df['cohort_start_date']+timedelta(days=post_observation_peroid))]
            # condition 3) Delete data after first_abnormal_date (Except when there is no first abnormal date.)
            # domain_df = domain_df[~(domain_df['first_abnormal_date']<domain_df['concept_date'])]
            # domain_df = domain_df[~(domain_df['first_abnormal_date']-domain_df['cohort_start_date']>timedelta(days=post_observation_peroid))]
            domain_df = domain_df.reset_index(drop=True)
            curr_len = len(domain_df)
            print('{} > {}'.format(prev_len, curr_len))
            return domain_df    

        meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60)
        drug_df = cohortConditionSetting(drug_df, pre_observation_period=60, post_observation_peroid=60)
        proc_df = cohortConditionSetting(proc_df, pre_observation_period=60, post_observation_peroid=60)
        cond_df = cohortConditionSetting(cond_df, pre_observation_period=60, post_observation_peroid=60)

        ### population + domain + concept
        meas_df = pd.merge(meas_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
        drug_df = pd.merge(drug_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
        proc_df = pd.merge(proc_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
        cond_df = pd.merge(cond_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")

        # In[ ]:
        # ***** setting first abnormal date *****
        # # ** hepatotoxicity (간독성) **

        if 'hepatotoxicity' == cfg['drug'][outcome_name]['ade'] :
            _3times = 3
            _2times = 2
            _1_5times = 1.5

            concept_id_AST = cfg['meas']["AST"]['@meas_concept_id']
            concept_id_ALT = cfg['meas']["ALT"]['@meas_concept_id']
            concept_id_ALP = cfg['meas']["ALP"]['@meas_concept_id']
            concept_id_TB = cfg['meas']["TB"]['@meas_concept_id']

            def extraction_of_past_abnormalities(domain_df, concept_id, value):
                n_prev_data = len(domain_df)
                n_prev_person = len(domain_df.person_id.unique())
                history_query = """(sex==1 and cohort_start_date>=concept_date and concept_id=={} and concept_value>{} and range_high > 0) or \
                    (sex==0 and cohort_start_date>=concept_date and concept_id=={} and concept_value>{} and range_high > 0)""" \
                    .format(concept_id, value, concept_id, value)
                historynormal_person = domain_df.query(history_query)
                print(history_query)
                n_post_data = len(historynormal_person)
                n_prev_person = len(historynormal_person.person_id.unique())
                # print(historynormal_person)
                print('{} > {}'.format(n_prev_data, n_post_data))
                print('{} > {}'.format(n_prev_person, n_prev_person))
                return historynormal_person

            past_abnormalities_AST = extraction_of_past_abnormalities(meas_df, concept_id_AST, value="range_high")
            past_abnormalities_ALT = extraction_of_past_abnormalities(meas_df, concept_id_ALT, value="range_high")
            past_abnormalities_ALP = extraction_of_past_abnormalities(meas_df, concept_id_ALP, value="range_high")
            past_abnormalities_TB = extraction_of_past_abnormalities(meas_df, concept_id_TB, value="range_high")

            past_abnormalities = set(past_abnormalities_AST.person_id.unique()) | \
                set(past_abnormalities_ALT.person_id.unique()) | \
                set(past_abnormalities_ALP.person_id.unique()) | \
                set(past_abnormalities_TB.person_id.unique())

            print("dropout : n = ", len(past_abnormalities))

            def extraction_of_abnormalities(domain_df, concept_id, value):
                n_prev_data = len(domain_df)
                n_prev_person = len(domain_df.person_id.unique())
                history_query = """(sex==1 and cohort_start_date<concept_date and concept_id=={} and concept_value>=({}) and range_high != 0) or \
                    (sex==0 and cohort_start_date<concept_date and concept_id=={} and concept_value>=({}) and range_high != 0)""" \
                    .format(concept_id, value, concept_id, value)
                historynormal_person = domain_df.query(history_query)
                print(history_query)
                n_post_data = len(historynormal_person)
                n_prev_person = len(historynormal_person.person_id.unique())
                # print(historynormal_person)
                print('{} > {}'.format(n_prev_data, n_post_data))
                print('{} > {}'.format(n_prev_person, n_prev_person))
                return historynormal_person

            sql_query_ALT = """(cohort_start_date<concept_date and concept_id=={} and concept_value>={} and range_high > 0)""" \
                    .format(concept_id_ALT, "range_high*5")
            sql_query_ALP = """(cohort_start_date<concept_date and concept_id=={} and concept_value>={} and range_high > 0)""" \
                    .format(concept_id_ALP, "range_high*2")
            sql_query_ALT_TB_1 = """(cohort_start_date<concept_date and concept_id=={} and concept_value>={} and range_high > 0)""" \
                    .format(concept_id_ALT, "range_high*3")
            sql_query_ALT_TB_2 = """(cohort_start_date<concept_date and concept_id=={} and concept_value>{} and range_high > 0)""" \
                    .format(concept_id_TB, "range_high*2")

            abnormalities_ALT_range = meas_df.query(sql_query_ALT)
            abnormalities_ALP_range = meas_df.query(sql_query_ALP)
            abnormalities_ALT_TB_1_range = meas_df.query(sql_query_ALT_TB_1)
            abnormalities_ALT_TB_2_range = meas_df.query(sql_query_ALT_TB_2)

            abnormalities = set(abnormalities_ALT_range.person_id.unique()) \
                | set(abnormalities_ALP_range.person_id.unique()) \
                | (set(abnormalities_ALT_TB_1_range.person_id.unique()) & set(abnormalities_ALT_TB_2_range.person_id.unique()))
            
            # In[]:
            print("abnormal : n = ", len(abnormalities))

            abnormalities_df = meas_df.query("cohort_start_date<concept_date")
            abnormalities_df = abnormalities_df[abnormalities_df["person_id"].isin(abnormalities)]
            abnormalities_df = abnormalities_df[["person_id", "concept_date"]]
            abnormalities_df = abnormalities_df.sort_values(by=["person_id", "concept_date"], axis=0, ascending=[True, True]).reset_index(drop=True)
            abnormalities_df = abnormalities_df.rename({"concept_date":"first_abnormal_date"}, axis=1)
            abnormalities_df = abnormalities_df.drop_duplicates(subset=['person_id'], keep = 'first')
            print(abnormalities_df)
            # population_df['label'] = (~population_df['first_abnormal_date'].isnull()).astype(int)

        # In[ ]:
        # **nephrotoxicity(신독성)**

        if 'nephrotoxicity' == cfg['drug'][outcome_name]['ade'] :
            _3times = 3
            _2times = 2
            _1_5times = 1.5

            concept_id_CR = cfg['meas']["CR"]['@meas_concept_id']

            def extraction_of_past_abnormalities(domain_df, concept_id, value):
                n_prev_data = len(domain_df)
                n_prev_person = len(domain_df.person_id.unique())
                history_query = """(sex==1 and cohort_start_date>=concept_date and concept_id=={} and concept_value>{}) or \
                    (sex==0 and cohort_start_date>=concept_date and concept_id=={} and concept_value>{})""" \
                    .format(concept_id, value, concept_id, value)
                historynormal_person = domain_df.query(history_query)
                print(history_query)
                n_post_data = len(historynormal_person)
                n_prev_person = len(historynormal_person.person_id.unique())
                # print(historynormal_person)
                print('{} > {}'.format(n_prev_data, n_post_data))
                print('{} > {}'.format(n_prev_person, n_prev_person))
                return historynormal_person

            past_abnormalities_creatinine = extraction_of_past_abnormalities(meas_df, concept_id_CR, value="range_high")

            past_abnormalities = set(past_abnormalities_creatinine.person_id.unique()) 

            print("dropout : n = ", len(past_abnormalities))

            def extraction_of_abnormalities(domain_df, concept_id, value):
                n_prev_data = len(domain_df)
                n_prev_person = len(domain_df.person_id.unique())
                history_query = """(sex==1 and cohort_start_date<concept_date and concept_id=={} and concept_value>=({})) or \
                    (sex==0 and cohort_start_date<concept_date and concept_id=={} and concept_value>=({}))""" \
                    .format(concept_id, value, concept_id, value)
                historynormal_person = domain_df.query(history_query)
                print(history_query)
                n_post_data = len(historynormal_person)
                n_prev_person = len(historynormal_person.person_id.unique())
                # print(historynormal_person)
                print('{} > {}'.format(n_prev_data, n_post_data))
                print('{} > {}'.format(n_prev_person, n_prev_person))
                return historynormal_person

            abnormalities_creatinine_value = extraction_of_abnormalities(meas_df, concept_id_CR, value="2.4")
            abnormalities = set(abnormalities_creatinine_value.person_id.unique())

            print("abnormal : n = ", len(abnormalities))

            abnormalities_df = abnormalities_creatinine_value[["person_id", "concept_date"]]
            abnormalities_df = abnormalities_df.sort_values(by=["person_id", "concept_date"], axis=0, ascending=[True, True]).reset_index(drop=True)
            abnormalities_df = abnormalities_df.rename({"concept_date":"first_abnormal_date"}, axis=1)
            abnormalities_df = abnormalities_df.drop_duplicates(subset=['person_id'], keep = 'first')
            print(abnormalities_df)

        # In[ ]:
        ### **outcome 마다 각 domain dataset 생성**

        if (len(past_abnormalities) > 0) :
            meas_df = meas_df[~meas_df["person_id"].isin(past_abnormalities)]
            drug_df = drug_df[~drug_df["person_id"].isin(past_abnormalities)]
            proc_df = proc_df[~proc_df["person_id"].isin(past_abnormalities)]
            cond_df = cond_df[~cond_df["person_id"].isin(past_abnormalities)]

        meas_df = pd.merge(meas_df, abnormalities_df, left_on=["person_id"], right_on=["person_id"], how="left").reset_index(drop=True)
        drug_df = pd.merge(drug_df, abnormalities_df, left_on=["person_id"], right_on=["person_id"], how="left").reset_index(drop=True)
        proc_df = pd.merge(proc_df, abnormalities_df, left_on=["person_id"], right_on=["person_id"], how="left").reset_index(drop=True)
        cond_df = pd.merge(cond_df, abnormalities_df, left_on=["person_id"], right_on=["person_id"], how="left").reset_index(drop=True)

        # In[]:
        meas_df = meas_df.rename({"first_abnormal_date_y":"first_abnormal_date"}, axis=1)
        drug_df = drug_df.rename({"first_abnormal_date_y":"first_abnormal_date"}, axis=1)
        proc_df = proc_df.rename({"first_abnormal_date_y":"first_abnormal_date"}, axis=1)
        cond_df = cond_df.rename({"first_abnormal_date_y":"first_abnormal_date"}, axis=1)

        meas_df['label'] = (~meas_df['first_abnormal_date'].isnull()).astype(int)
        drug_df['label'] = (~drug_df['first_abnormal_date'].isnull()).astype(int)
        proc_df['label'] = (~proc_df['first_abnormal_date'].isnull()).astype(int)
        cond_df['label'] = (~cond_df['first_abnormal_date'].isnull()).astype(int)


        # In[ ]:
        # ** Get only used dates **
        def cohortConditionSetting(domain_df, pre_observation_period, post_observation_peroid):
            from datetime import timedelta
            prev_len = len(domain_df)
            domain_df['cohort_start_date'] = pd.to_datetime(domain_df['cohort_start_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            domain_df['first_abnormal_date'] = pd.to_datetime(domain_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            domain_df['concept_date'] = pd.to_datetime(domain_df['concept_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            # condition 1) Select patients with first adverse events within 2 months of cohort initiation.
            domain_df = domain_df[(domain_df['cohort_start_date']<=domain_df['concept_date']+timedelta(days=pre_observation_period))]
            # condition 2) Delete data before the cohort start date.
            domain_df = domain_df[(domain_df['concept_date']<=domain_df['cohort_start_date']+timedelta(days=post_observation_peroid))]
            # condition 3) Delete data after first_abnormal_date (Except when there is no first abnormal date.)
            # domain_df = domain_df[~(domain_df['first_abnormal_date']<domain_df['concept_date'])]
            domain_df = domain_df[~(domain_df['first_abnormal_date']-domain_df['cohort_start_date']>timedelta(days=post_observation_peroid))]
            # domain_df["first_abnormal_date"] = domain_df["first_abnormal_date"].fillna('1970-01-01')
            domain_df = domain_df.reset_index(drop=True)
            curr_len = len(domain_df)
            print('{} > {}'.format(prev_len, curr_len))
            return domain_df    

        meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60)
        drug_df = cohortConditionSetting(drug_df, pre_observation_period=60, post_observation_peroid=60)
        proc_df = cohortConditionSetting(proc_df, pre_observation_period=60, post_observation_peroid=60)
        cond_df = cohortConditionSetting(cond_df, pre_observation_period=60, post_observation_peroid=60)

        meas_df['concept_domain'] = 'meas'
        drug_df['concept_domain'] = 'drug'
        proc_df['concept_domain'] = 'proc'
        cond_df['concept_domain'] = 'cond'

        meas_df.to_csv('{}/{}_meas_df.txt'.format(output_dir, outcome_name),index=False)
        drug_df.to_csv('{}/{}_drug_df.txt'.format(output_dir, outcome_name),index=False)
        proc_df.to_csv('{}/{}_proc_df.txt'.format(output_dir, outcome_name),index=False)
        cond_df.to_csv('{}/{}_cond_df.txt'.format(output_dir, outcome_name),index=False)

        # In[ ]:
        all_domain_vars_df = pd.concat([meas_df, drug_df, proc_df, cond_df], axis=0, ignore_index=True)
        n_label1 = len(all_domain_vars_df[all_domain_vars_df['label']==1].person_id.unique())
        n_label0 = len(all_domain_vars_df[all_domain_vars_df['label']==0].person_id.unique())
        print('label 1 : ', n_label1)
        print('label 0 : ', n_label0)


        # In[ ]:
        out = open('{}/output.txt'.format(output_result_dir),'a')
        out.write(str(outcome_name) + '///' )
        out.write(str(n_label1) + '///')
        out.write(str(n_label0) + '///')
        out.close()

    except :
        traceback.print_exc()
        log.error(traceback.format_exc())

conn.close()

# %%
