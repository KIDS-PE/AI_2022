#!/usr/bin/env python
# coding: utf-8

'''
preprocessing (LSTM)
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
from _utils.preprocessing_lstm import *
from _utils.customlogger import customlogger as CL

# In[ ]:
# ** loading config **
with open('./../{}'.format("config.json")) as file:
    cfg = json.load(file)
with open('./../{}'.format("config_params.json")) as file:
    params = json.load(file)

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
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        # In[ ]:
        log.debug("{}".format(outcome_name))
        # input file path
        importsql_output_dir = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))
        # output file path
        output_dir = pathlib.Path('{}/data/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)
        # output file path (features)
        output_result_dir = pathlib.Path('{}/result/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

        # In[ ]:
        # @load data
        meas_df = pd.read_csv('{}/{}_meas_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
        drug_df = pd.read_csv('{}/{}_drug_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
        proc_df = pd.read_csv('{}/{}_proc_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
        cond_df = pd.read_csv('{}/{}_cond_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)

        # @fill concept_value
        drug_df['concept_value'] = 1 # temp code
        proc_df['concept_value'] = 1
        cond_df['concept_value'] = 1

        # @use only necessary columns
        common_cols = ['person_id', 'age', 'sex',     'cohort_start_date', 'first_abnormal_date', 'concept_date',     'concept_id', 'concept_name', 'concept_value', 'concept_domain', 'label']

        meas_df = meas_df[common_cols]
        drug_df = drug_df[common_cols]
        proc_df = proc_df[common_cols]
        cond_df = cond_df[common_cols]

        print(len(meas_df), len(drug_df), len(proc_df), len(cond_df), (len(meas_df) + len(drug_df) + len(proc_df) + len(cond_df)))

        # @valid data processing for cohorts.
        meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60)
        drug_df = cohortConditionSetting(drug_df, pre_observation_period=60, post_observation_peroid=60)
        proc_df = cohortConditionSetting(proc_df, pre_observation_period=60, post_observation_peroid=60)
        cond_df = cohortConditionSetting(cond_df, pre_observation_period=60, post_observation_peroid=60)

        all_domain_vars_df = pd.concat([meas_df, drug_df, proc_df, cond_df], axis=0, ignore_index=True)
        print('label 1 : ', len(all_domain_vars_df[all_domain_vars_df['label']==1].person_id.unique()))
        print('label 0 : ', len(all_domain_vars_df[all_domain_vars_df['label']==0].person_id.unique()))

        # def average_duration_of_adverse_events(df):
        #     df = df[['person_id', 'cohort_start_date', 'first_abnormal_date']].drop_duplicates() #.subject_id.unique()
        #     df['c_f'] = df['first_abnormal_date'] - df['cohort_start_date']
        #     print(df['c_f'].describe())
        #     return df['c_f'].mean().days

        # ndays = average_duration_of_adverse_events(cond_df)
        # print(ndays)

        # person_df = meas_df[["person_id", "label"]].drop_duplicates()
        # print(person_df.label.value_counts())
        # person_df = drug_df[["person_id", "label"]].drop_duplicates()
        # print(person_df.label.value_counts())
        # person_df = proc_df[["person_id", "label"]].drop_duplicates()
        # print(person_df.label.value_counts())
        # person_df = cond_df[["person_id", "label"]].drop_duplicates()
        # print(person_df.label.value_counts())
        
        # ---------------------- check features ----------------------------
        concept_list = []
        nCaseInTotal = len(all_domain_vars_df.loc[all_domain_vars_df['label']==1,'person_id'].unique())
        nControlInTotal =len(all_domain_vars_df.loc[all_domain_vars_df['label']==0,'person_id'].unique())

        meas_df = meas_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        drug_df = drug_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        cond_df = cond_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        proc_df = proc_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)

        meas_concept_df = meas_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        drug_concept_df = drug_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        cond_concept_df = cond_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
        proc_concept_df = proc_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)

        meas_concept_df['concept_domain'] = 'meas'
        drug_concept_df['concept_domain'] = 'drug'
        cond_concept_df['concept_domain'] = 'proc'
        proc_concept_df['concept_domain'] = 'cond'
        
        all_domain_concept_df = pd.concat([meas_concept_df, drug_concept_df, cond_concept_df, proc_concept_df], axis=0, ignore_index=True)
        all_domain_concept_df.to_csv('{}/{}_feature_2.csv'.format(output_result_dir, outcome_name), header=True, index=True)
        # -------------------------------------------------------------------
        
        # @variable selection
        meas_vars_df = variant_selection_paired_t_test(meas_df)
        drug_vars_df = variant_selection_mcnemar(drug_df)
        proc_vars_df = variant_selection_mcnemar(proc_df)
        cond_vars_df = variant_selection_mcnemar(cond_df)

        # @variable selection (Top 30 based on p Value)
        #pd.options.display.precision = 3
        meas_vars_df = meas_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
        drug_vars_df = drug_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
        cond_vars_df = cond_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
        proc_vars_df = proc_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
        print(len(meas_vars_df), len(drug_vars_df), len(proc_vars_df), len(cond_vars_df))
        
        meas_vars_df['concept_domain'] = 'meas'
        drug_vars_df['concept_domain'] = 'drug'
        cond_vars_df['concept_domain'] = 'proc'
        proc_vars_df['concept_domain'] = 'cond'
        all_domain_vars_df = pd.concat([meas_vars_df, drug_vars_df, cond_vars_df, proc_vars_df], axis=0, ignore_index=True)
        # @variable selection (save)
        all_domain_vars_df.to_csv('{}/{}_feature.csv'.format(output_result_dir, outcome_name), header=True, index=True)
        # all_domain_vars_df = pd.read_csv('{}/{}_{}_feature.csv'.format(output_result_dir, outcome_name), index_col=False) #check

        # @Extract only selected concepts from data frame
        def extractSelectedConceptID(domain_df, concept_id_list):
            extract_domain_df = domain_df[domain_df['concept_id'].isin(concept_id_list)]
            print(len(concept_id_list), len(domain_df), len(extract_domain_df))
            return extract_domain_df

        meas_df2 = extractSelectedConceptID(meas_df, meas_vars_df.concept_id.unique())
        drug_df2 = extractSelectedConceptID(drug_df, drug_vars_df.concept_id.unique())
        proc_df2 = extractSelectedConceptID(proc_df, proc_vars_df.concept_id.unique())
        cond_df2 = extractSelectedConceptID(cond_df, cond_vars_df.concept_id.unique())

        # meas_df2 = extractSelectedConceptID(meas_df2, meas_common_features.concept_id.unique())
        # drug_df2 = extractSelectedConceptID(drug_df2, drug_common_features.concept_id.unique())
        # proc_df2 = extractSelectedConceptID(proc_df2, proc_common_features.concept_id.unique())
        # cond_df2 = extractSelectedConceptID(cond_df2, cond_common_features.concept_id.unique())

        all_domain_df = pd.concat([meas_df2, drug_df2, proc_df2, cond_df2], axis=0, ignore_index=True)
        # all_domain_df.drop(all_domain_df[all_domain_df['concept_domain']=='drug'].index, inplace=True)

        # ## @성향점수 매칭 (Propensity Score matching)
        # label1_df = all_domain_df.drop_duplicates(['person_id'])[all_domain_df['label']==1][['person_id', 'age']]
        # label0_df = all_domain_df.drop_duplicates(['person_id'])[all_domain_df['label']==0][['person_id', 'age']]

        # matched_df = get_matching_pairs(label1_df['age'], label0_df['age'], scaler=False)
        # person_id_list = set(label0_df.loc[matched_df.index].person_id) | set(label1_df.person_id)

        # def extract_selected_person_ids(domain_df, subject_id_list):
        #     extract_domain_df = domain_df[domain_df['person_id'].isin(subject_id_list)]
        #     print(len(subject_id_list), len(domain_df), len(extract_domain_df))
        #     return extract_domain_df

        # all_domain_df = extract_selected_person_ids(all_domain_df, person_id_list).reset_index(drop=True)
        ## @성향점수 매칭 end
        
        # # @test : 
        # averageDurationOfAE = average_duration_of_adverse_events(all_domain_df)
        # print(averageDurationOfAE)

        pivot_data = pivotting(all_domain_df)
        # pivot_data = pivot_data.query("concept_date <= cohort_start_date")
        # pivot_data = pivot_data.sort_values(by=["person_id", "concept_date"], axis=0, ascending=[True, False]).reset_index(drop=True)
        # pivot_data = pivot_data.drop_duplicates(subset=['person_id'], keep = 'first')
        # pivot_data = pivot_data.fillna(0)

        # # temp 
        # pivot_data.to_csv('{}/{}_pivot_data.csv'.format(output_features_dir, outcome_name), header=True, index=True)
        drop_cols = []
        for col in pivot_data.columns:
            if (len(pivot_data[pivot_data[col].notnull()])/len(pivot_data[col]) < 0.3):
                drop_cols.append(col)
        print(drop_cols)
        pivot_data = pivot_data.drop(drop_cols, axis='columns')

        domain_ids={}
        domain_ids['meas'] = np.setdiff1d(meas_df2.concept_id.unique(), drop_cols)
        domain_ids['drug'] = np.setdiff1d(drug_df2.concept_id.unique(), drop_cols)
        domain_ids['proc'] = np.setdiff1d(proc_df2.concept_id.unique(), drop_cols)
        domain_ids['cond'] = np.setdiff1d(cond_df2.concept_id.unique(), drop_cols)

        # -------- time series data ---------
        interpolate_df = day_sequencing_interpolate(pivot_data, domain_ids, OBP=params["windowsize"])

        label_1 = interpolate_df[interpolate_df['label']==1]
        label_0 = interpolate_df[interpolate_df['label']==0]

        rolled_label1_d = shift_rolling_window(label_1, OBP=params["windowsize"], nShift=params["shift"], uid_index=1)
        rolled_label0_d = label_0_fitting(label_0, OBP=params["windowsize"], nShift=params["shift"], uid_index=(rolled_label1_d.unique_id.max()+1))

        # label 0 + label 1
        concat_df = pd.concat([rolled_label1_d, rolled_label0_d], sort=False)
        concat_df = concat_df.sort_values(['unique_id', 'concept_date'])
        # -------- time series data ---------

        # Normalization (Min/Max Scalar)
        concat_df = normalization(concat_df)
        concat_df = concat_df.dropna(axis=1)

        # columns name : concept_id > concept_name
        concept_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_name))
        concat_df = concat_df.rename(concept_dict, axis='columns')

        # Save File
        concat_df.to_csv('{}/{}.txt'.format(output_dir, outcome_name), index=False, float_format='%g')

        output={}
        output['meas_whole_var'] = len(meas_df.concept_id.unique())
        output['drug_whole_var'] = len(drug_df.concept_id.unique())
        output['proc_whole_var'] = len(proc_df.concept_id.unique())
        output['cond_whole_var'] = len(cond_df.concept_id.unique())
        output['meas_selected_var'] = len(domain_ids['meas'])
        output['drug_selected_var'] = len(domain_ids['drug'])
        output['proc_selected_var'] = len(domain_ids['proc'])
        output['cond_selected_var'] = len(domain_ids['cond'])
        output['nPatient_label1'] = len(concat_df[concat_df["label"] == 1])
        output['nPatient_label0'] = len(concat_df[concat_df["label"] == 0])

        # print
        print(output['meas_whole_var'], output['meas_selected_var'])
        print(output['drug_whole_var'], output['drug_selected_var'])
        print(output['proc_whole_var'], output['proc_selected_var'])
        print(output['cond_whole_var'], output['cond_selected_var'])

        out = open('{}/output.txt'.format(output_result_dir),'a')

        out.write(str(outcome_name) + '///' )
        out.write(str(output['meas_whole_var']) + '///')
        out.write(str(output['meas_selected_var']) + '///')
        out.write(str(output['drug_whole_var']) + '///')
        out.write(str(output['drug_selected_var']) + '///')
        out.write(str(output['proc_whole_var']) + '///')
        out.write(str(output['proc_selected_var']) + '///')
        out.write(str(output['cond_whole_var']) + '///')
        out.write(str(output['cond_selected_var']) + '///')
        out.write(str(output['nPatient_label1']) + '///')
        out.write(str(output['nPatient_label0']) + '\n')
        out.close()

    except :
        traceback.print_exc()
        log.error(traceback.format_exc())
# %%
