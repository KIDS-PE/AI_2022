#!/usr/bin/env python
# coding: utf-8

'''
feature selection
'''
# In[]:
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
from _utils.preprocessing_xgboost import *
from _utils.customlogger import customlogger as CL
# In[]:
# ** loading config **
with open('./../{}'.format("config.json")) as file:
    cfg = json.load(file)
# In[]:
# ** loading info **
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent
current_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(os.path.abspath('')))[0]
# In[]:
# **create Logger**
log = CL("custom_logger")
pathlib.Path.mkdir(pathlib.Path('{}/_log/'.format(parent_dir)), mode=0o777, parents=True, exist_ok=True)
log = log.create_logger(file_name="../_log/{}.log".format(curr_file_name), mode="a", level="DEBUG")  
log.debug('start {}'.format(curr_file_name))
# In[]:
def intersect(a, b):
    return list(set(a) & set(b))

def getPairedTTest(baseline_df, abnormal_df, concept_list):
    baseline_df = baseline_df[baseline_df['label']==1]
    abnormal_df = abnormal_df[abnormal_df['label']==1]
    import scipy.stats
    selected_var_df = pd.DataFrame()
    concept_set = list(set(baseline_df.columns) & set(abnormal_df.columns) & set(concept_list))
    # print(len(concept_set), concept_set)
    for concept in concept_set:
        # print(abnormal_df[concept].mean(), baseline_df[concept].mean())
        statistic, pvalue = scipy.stats.ttest_ind(abnormal_df[concept], baseline_df[concept], equal_var=False, nan_policy='omit')
        label_1_before = len(baseline_df[concept].dropna()) 
        label_1_after = len(abnormal_df[concept].dropna()) 
        label_1_before_mean = baseline_df[concept].dropna().mean() 
        label_1_after_mean = abnormal_df[concept].dropna().mean()
        # print(concept, pvalue)
        if statistic>1 and pvalue<0.05 :
            # print(concept)
            var_temp = {}
            var_temp['concept_id'] = concept
            var_temp['pvalue'] = pvalue
            var_temp['label_1_before'] = label_1_before
            var_temp['label_1_after'] = label_1_after
            var_temp['label_1_before_mean'] = label_1_before_mean
            var_temp['label_1_after_mean'] = label_1_after_mean
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

def getMcnemarTest(baseline_df, abnormal_df, concept_list):
    import scipy.stats
    selected_var_df = pd.DataFrame()
    concept_set = list(set(baseline_df.columns) & set(abnormal_df.columns) & set(concept_list))
    # print(len(concept_set), concept_set)
    for concept in concept_set:
        label_0_before = len(baseline_df[(baseline_df['label']==0) & (baseline_df[concept]==1)])
        label_1_before = len(baseline_df[(baseline_df['label']==1) & (baseline_df[concept]==1)])
        label_0_after = len(abnormal_df[(abnormal_df['label']==0) & (abnormal_df[concept]==1)]) 
        label_1_after = len(abnormal_df[(abnormal_df['label']==1) & (abnormal_df[concept]==1)]) 
        arr_before = np.array([label_1_before, label_0_before])
        arr_after = np.array([label_1_after, label_0_after])
        table = np.vstack([arr_before, arr_after]) # vertical stack
        table = np.transpose(table)             # trans pose
        result = mcnemar(table, exact=True) # 샘플 수<25 일 경우 mcnemar(table, exact=False, correction=True)
        if result.pvalue < 0.05 :
            # print(concept)
            var_temp = {}
            var_temp['concept_id'] = concept
            var_temp['pvalue'] = result.pvalue
            var_temp['label_0_before'] = label_0_before
            var_temp['label_0_after'] = label_0_after
            var_temp['label_1_before'] = label_1_before
            var_temp['label_1_after'] = label_1_after
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

def average_duration_of_adverse_events(df):
    df = df[['person_id', 'cohort_start_date', 'first_abnormal_date']].drop_duplicates() #.subject_id.unique()
    df['c_f'] = df['first_abnormal_date'] - df['cohort_start_date']
    # print(df['c_f'].describe())
    return df['c_f'].mean().days

def make_pivot(df):
    if df.empty:
        return pd.DataFrame()
    print("person_id(count) : ", df.person_id.nunique(), "concept_name(count) : ", df.concept_name.nunique())
    df = df.sort_values(by=['person_id', 'concept_id', 'concept_date'], axis=0, ascending=True)
    df['first_abnormal_date'] = pd.to_datetime(df['first_abnormal_date']).fillna(pd.to_datetime('1900-01-01'))
    last_record_df = df.groupby(by=['person_id', 'concept_id']).apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    def subtract(x, y):
        return [item for item in x if item not in set(y)]
    pivot_cols = subtract(last_record_df.columns, ['concept_name', 'concept_date', 'concept_id', 'concept_value', 'concept_domain'])
    pivot_df = pd.pivot_table(data = last_record_df, index = pivot_cols, columns='concept_id', values='concept_value').reset_index()
    return pivot_df

def impute_conditional_data(df, concept_ids):
    cols = list(set(df.columns)&set(concept_ids))
    df[cols] = df[cols].fillna(df[cols].median())
    return df
    
def impute_binary_data(df, concept_ids):
    cols = list(set(df.columns)&set(concept_ids))
    df[cols] = df[cols].fillna(0)
    return df

def normalization_Robust(df):
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 

def normalization_std(df):
    from sklearn.preprocessing import StandardScaler
    transformer = StandardScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 

def normalization_minmax(df):
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 
# In[]:
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel, SelectPercentile, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
%matplotlib inline

def add_column_concept_name(df, concept_id_name_dict):
    df['concept_name'] = df.apply(lambda x: concept_id_name_dict[x.concept_id] if x.concept_id in concept_id_name_dict.keys() else x.concept_id, axis = 1)
    return df

def add_column_concept_domain(df, concept_id_domain_dict):
    df['concept_domain'] = df.apply(lambda x: concept_id_domain_dict[x.concept_id] if x.concept_id in concept_id_domain_dict.keys() else 'common', axis = 1)
    print(df.concept_domain.value_counts())
    return df

def make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict):
    if len(selected_features) < 1:
        return pd.DataFrame()
        
    df = pd.DataFrame(selected_features, columns =['concept_id'])
    df = add_column_concept_name(df, concept_id_name_dict)
    df = add_column_concept_domain(df, concept_id_domain_dict)
    print(len(df.concept_id.unique()))
    return df

def write_file_method(df, dir, name, method):
    if df.empty:
        return False
    full_file_path = pathlib.Path('{}/{}_{}.csv'.format(dir, name, method))
    df.to_csv(full_file_path, index=False, float_format='%g')
    return True

def read_files_method(dir, name, method):
    full_file_path = pathlib.Path('{}/{}_{}.csv'.format(dir, name, method))
    if not pathlib.Path.exists(full_file_path):
        return pd.DataFrame()
    df = pd.read_csv(full_file_path)
    return df

def read_files_all_methods(dir, name):
    methods = ['statistics', 'VT', 'KBest', 'percentile', 'ExtraTrees', 'lasso_0_1', 'lasso_0_0_1', 'mutual']
    concat_df = pd.DataFrame()
    for method in methods:
        method_df = read_files_method(dir, name, method)
        if method_df.empty:
            continue
        method_df['method'] = method
        concat_df = pd.concat([concat_df, method_df], axis=0)
    return concat_df

def resumetable(df):
    print(f'data frame shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['data_type'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'feature'})
    summary['n_values'] = df.notnull().sum().values
    summary['n_missingvalues'] = df.isnull().sum().values
    summary['n_missingrate'] = df.isnull().sum().values/len(df)
    summary['n_eigenvalues'] = df.nunique().values
    return summary

def getxydata(df):
    x_df = df.drop(['person_id', 'cohort_start_date', 'first_abnormal_date', 'label'], axis=1) 
    # x_df = (x_df-x_df.min())/(x_df.max()-x_df.min()) # normalize
    x_data = x_df.to_numpy()
    y_data = df['label'].to_numpy()
    cols = x_df.columns
    return x_data, y_data, cols

def concat_all_methods(feature_selection_method_df_dict):
    concat_df = pd.DataFrame()
    for method in feature_selection_method_df_dict.keys():
        method_df = feature_selection_method_df_dict[method]
        if method_df.empty:
            continue
        method_df['method'] = method
        concat_df = pd.concat([concat_df, method_df], axis=0)
    return concat_df

def merge_summary_table_df(summary_df, concat_df):
    pivot_df = concat_df[['concept_id', 'concept_name', 'concept_domain', 'method']]
    pivot_df['value'] = 1
    pivot_df = pd.pivot_table(data=pivot_df, columns='method', index=['concept_id', 'concept_name', 'concept_domain'], values='value', fill_value=0).reset_index()
    pivot_df['total'] = pivot_df[list(set(concat_df.method.unique()) & set(pivot_df.columns))].sum(axis=1)
    pivot_df
    pivot_df.concept_id = pivot_df.concept_id.apply(lambda _: str(_).replace('.0',''))
    summary_df.concept_id = summary_df.concept_id.apply(lambda _: str(_).replace('.0',''))
    pivot_join_df = pd.merge(left=pivot_df, right=summary_df, left_on=['concept_id'], right_on=['concept_id'], how='left')
    # old_columns = pivot_join_df.columns.to_list()
    # new_columns = ['{}_{}'.format(col,hospital[0]) for col in pivot_join_df.columns]
    # pivot_join_df.rename(dict(zip(old_columns, new_columns)), axis=1, inplace=True)
    return pivot_join_df
# In[ ]:
def runTask(outcome_name):
    # outcome_name = 'Cisplatin'
    log.debug("{}".format(outcome_name))

    importsql_output_dir    = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))
    output_data_dir         = pathlib.Path('{}/data/{}/feature_selection/{}/'.format(parent_dir, current_date, outcome_name))
    output_result_dir       = pathlib.Path('{}/result/{}/feature_selection/{}/'.format(parent_dir, current_date, outcome_name))
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

    # In[ ]: @load data
    meas_df = pd.read_csv('{}/{}_meas_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
    drug_df = pd.read_csv('{}/{}_drug_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
    proc_df = pd.read_csv('{}/{}_proc_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)
    cond_df = pd.read_csv('{}/{}_cond_df.txt'.format(importsql_output_dir, outcome_name), low_memory=False)

    # In[ ]: @fill concept_value
    drug_df['concept_value'] = 1 # temp code
    proc_df['concept_value'] = 1
    cond_df['concept_value'] = 1

    # In[ ]: @use only necessary columns
    common_cols = ['person_id', 'age', 'sex', 'cohort_start_date', 'first_abnormal_date', 'concept_date', 'concept_id', 'concept_name', 'concept_value', 'concept_domain', 'label']

    meas_df = meas_df[common_cols]
    drug_df = drug_df[common_cols]
    proc_df = proc_df[common_cols]
    cond_df = cond_df[common_cols]

    log.info("[nData] m : {} d : {} p : {}  c : {} all : {}".format(len(meas_df), len(drug_df), len(proc_df), len(cond_df), (len(meas_df) + len(drug_df) + len(proc_df) + len(cond_df))))

    # In[ ]: @Remove feature used in outcome define
    drug_name = outcome_name
    drug_concept_ids_excluded = map(int,cfg['drug'][drug_name]['@drug_concept_set'].split(','))
    drug_df = drug_df.loc[~drug_df.concept_id.isin(drug_concept_ids_excluded)]
    meas_concept_ids_excluded = map(int,[cfg['meas'][meas_name]['@meas_concept_id'] for meas_name in cfg['meas']])
    meas_df = meas_df.loc[~meas_df.concept_id.isin(meas_concept_ids_excluded)]

    # In[ ]: @valid data processing for cohorts.
    meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60)
    drug_df = cohortConditionSetting(drug_df, pre_observation_period=60, post_observation_peroid=60)
    proc_df = cohortConditionSetting(proc_df, pre_observation_period=60, post_observation_peroid=60)
    cond_df = cohortConditionSetting(cond_df, pre_observation_period=60, post_observation_peroid=60)

    ndays = average_duration_of_adverse_events(cond_df)
    log.debug('average_duration_of_adverse_events : {}'.format(ndays))

    all_domain_df = pd.concat([meas_df, drug_df, proc_df, cond_df], axis=0, ignore_index=True)
    all_domain_baseline_df = all_domain_df.query('cohort_start_date >= concept_date')

    if all_domain_df.empty:
        return

    all_domain_pivot_df = make_pivot(all_domain_df)
    all_domain_pivot_baseline_df = make_pivot(all_domain_baseline_df)

    summary_df = resumetable(all_domain_pivot_df)
    write_file_method(summary_df, output_result_dir, outcome_name, 'summary')

    concept_id_name_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_name))
    concept_id_domain_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_domain))

    feature_selection_method_df_dict = {}

    # In[ ] @ Feature_Selection 1 : statistics method
    meas_concept_ids = list(set(all_domain_pivot_df.columns)&(set(meas_df.concept_id)))
    drug_concept_ids = list(set(all_domain_pivot_df.columns)&(set(drug_df.concept_id)))
    proc_concept_ids = list(set(all_domain_pivot_df.columns)&(set(proc_df.concept_id)))
    cond_concept_ids = list(set(all_domain_pivot_df.columns)&(set(cond_df.concept_id)))

    selected_features_with_t_test_df = getPairedTTest(all_domain_pivot_baseline_df, all_domain_pivot_df, meas_concept_ids)
    selected_features_with_mcnemar_df = getMcnemarTest(all_domain_pivot_baseline_df, all_domain_pivot_df, drug_concept_ids + cond_concept_ids + proc_concept_ids)

    selected_features_df = pd.concat([selected_features_with_t_test_df, selected_features_with_mcnemar_df], axis=0)
    if not selected_features_df.empty:
        selected_features_df.concept_id = selected_features_df.concept_id.astype(np.object)
    selected_features_df = add_column_concept_name(selected_features_df, concept_id_name_dict)
    selected_features_df = add_column_concept_domain(selected_features_df, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'statistics')
    feature_selection_method_df_dict['statistics'] = selected_features_df

    len(all_domain_df.concept_id.unique()), len(all_domain_baseline_df.concept_id.unique())

    # In[ ] @ imputation missing data
    all_domain_pivot_df = impute_conditional_data(all_domain_pivot_df, meas_concept_ids)
    all_domain_pivot_df = impute_binary_data(all_domain_pivot_df, drug_concept_ids + proc_concept_ids + cond_concept_ids)

    meas_concept_ids = list(set(all_domain_pivot_baseline_df.columns)&(set(meas_df.concept_id)))
    drug_concept_ids = list(set(all_domain_pivot_baseline_df.columns)&(set(drug_df.concept_id)))
    proc_concept_ids = list(set(all_domain_pivot_baseline_df.columns)&(set(proc_df.concept_id)))
    cond_concept_ids = list(set(all_domain_pivot_baseline_df.columns)&(set(cond_df.concept_id)))

    all_domain_pivot_baseline_df = impute_conditional_data(all_domain_pivot_baseline_df, meas_concept_ids)
    all_domain_pivot_baseline_df = impute_binary_data(all_domain_pivot_baseline_df, drug_concept_ids + proc_concept_ids + cond_concept_ids)

    X_total, y_total, cols = getxydata(all_domain_pivot_df)
    X_total = normalization_minmax(X_total)

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=1, stratify=y_total) 

    # check for nan / infinite value 
    np.argwhere(np.isnan(X_train)), np.argwhere(np.isinf(X_train))

    # In[ ] @ Feature_Selection 2 : VarianceThreshold
    selector = VarianceThreshold(1e-3)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'VT')
    feature_selection_method_df_dict['VT'] = selected_features_df

    # In[ ] @ Feature_Selection 3 : SelectPercentile
    selector = SelectPercentile(chi2, percentile=3) # now select features based on top 10 percentile
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'percentile')
    feature_selection_method_df_dict['percentile'] = selected_features_df

    # In[ ] @ Feature_Selection 4 : SelectKBest
    selector = SelectKBest(score_func=chi2, k=50)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'KBest')
    feature_selection_method_df_dict['KBest'] = selected_features_df

    # In[ ] @ Feature_Selection 5 : ExtraTreesClassifier
    treebasedclf = ExtraTreesClassifier(n_estimators=50)
    treebasedclf = treebasedclf.fit(X_train, y_train)
    model = SelectFromModel(treebasedclf, prefit=True)
    X_train_sel = model.transform(X_train)
    print(X_train.shape, X_train_sel.shape)
    selected_features = model.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'ExtraTrees')
    feature_selection_method_df_dict['ExtraTrees'] = selected_features_df

    # In[ ] @ Feature_Selection 6 : Lasso (1) > alpha = 0.1
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    # print(lasso.coef_)
    importance = np.abs(lasso.coef_)
    selected_features = np.array(cols)[importance > 0]
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)

    write_file_method(selected_features_df, output_result_dir, outcome_name, 'lasso_0_1')
    feature_selection_method_df_dict['lasso_0_1'] = selected_features_df

    # In[ ] @ Feature_Selection 7 : Lasso (2) > alpha = 0.0.1
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    # print(lasso.coef_)
    importance = np.abs(lasso.coef_)
    selected_features = np.array(cols)[importance > 0]
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'lasso_0_0_1')
    feature_selection_method_df_dict['lasso_0_0_1'] = selected_features_df

    # In[ ] @ Feature_Selection 8 : mutual_info_classif
    importances = mutual_info_classif(X_total, y_total, discrete_features='auto')
    threshold = 0.001
    selected_features = np.array(cols)[importance > threshold]
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'mutual')
    feature_selection_method_df_dict['mutual'] = selected_features_df

    # In[ ] @ all methods concatenate
    concat_df = concat_all_methods(feature_selection_method_df_dict)
    write_file_method(concat_df, output_result_dir, outcome_name, 'all_methods')
    summary_concat_df = merge_summary_table_df(summary_df, concat_df)
    write_file_method(summary_concat_df, output_result_dir, outcome_name, 'total')
    
# In[]:
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        runTask(outcome_name)        
    except :
        traceback.print_exc()
        log.error(traceback.format_exc())
