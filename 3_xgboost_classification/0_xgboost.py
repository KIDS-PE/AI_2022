#!/usr/bin/env python
# coding: utf-8

'''
### XGBoost
'''
# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib
sys.path.append("..")

import traceback
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from _utils.model_estimation import *
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
# **create Logger**
log = CL("custom_logger")
pathlib.Path.mkdir(pathlib.Path('{}/_log/'.format(parent_dir)), mode=0o777, parents=True, exist_ok=True)
log = log.create_logger(file_name="../_log/{}.log".format(curr_file_name), mode="a", level="DEBUG")  
log.debug('start {}'.format(curr_file_name))

# In[4]:
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        log.debug('drug : {}'.format(outcome_name))
        ps_data_dir = pathlib.Path('{}/data/{}/preprocess_xgboost/{}/'.format(parent_dir, current_date, outcome_name))
        output_result_dir = pathlib.Path('{}/result/{}/xgboost/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

        concat_df = pd.read_csv('{}/{}.txt'.format(ps_data_dir, outcome_name), index_col=False)

        concat_df['cohort_start_date'] = pd.to_datetime(concat_df['cohort_start_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        concat_df['first_abnormal_date'] = pd.to_datetime(concat_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        concat_df['concept_date'] = pd.to_datetime(concat_df['concept_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
            
        # concat_df['duration'] = (concat_df['concept_date']-concat_df['cohort_start_date']).dt.days+1
        concat_df = concat_df.drop(['person_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)

        ### @change column name ; column에 json파일 구분자가 들어가면 plot을 그리지 못함. 
        import re
        concat_df.columns = concat_df.columns.str.translate("".maketrans({"[":"(", "]":")"}))
        concat_df = concat_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_ /()]+', '', x))
        concat_df.columns

        ### @환자수 확인
        print("label_1 : ",len(concat_df[concat_df["label"] == 1]))
        print("label_0 : ",len(concat_df[concat_df["label"] == 0]))

        ### @x, y데이터 분할 
        def split_x_y_data(df) :
            y_data = df['label'].T.reset_index(drop=True) 
            x_data = df.drop('label', axis=1)
            new_col = x_data.columns
            return x_data, y_data, new_col

        x_data, y_data, new_col = split_x_y_data(concat_df)

        ### @train/test dataset 구분 
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1, stratify=y_data)

        print("data  : ", x_data.shape, y_data.shape)
        print("train : ", x_train.shape, y_train.shape)
        print("test  : ", x_test.shape, y_test.shape)
        
        scale_weight = int(len(concat_df[concat_df["label"] == 0])/len(concat_df[concat_df["label"] == 1]))

        dtrain = xgb.DMatrix(data=x_train , label=y_train) 
        dtest = xgb.DMatrix(data=x_test , label=y_test)

        params = { 'max_depth':5, 'learning_rate': 0.01, 'objective':'binary:logistic', 'eval_metric':'logloss', 'scale_pos_weight': scale_weight}
        num_rounds = 100
        # train 데이터 셋은 ‘train’ , evaluation(test) 데이터 셋은 ‘eval’ 로 명기합니다. 
        wlist = [(dtrain,'train'),(dtest,'eval')]
        # 하이퍼 파라미터와 early stopping 파라미터를 train( ) 함수의 파라미터로 전달 
        class_weight= class_balance_weight(output_result_dir, outcome_name, y_train)
            
        xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, early_stopping_rounds=25, evals=wlist)
        pred_probs = xgb_model.predict(dtest)

        # 예측 확률이 0.5 보다 크면 1 , 그렇지 않으면 0 으로 예측값 결정하여 List 객체인 preds 에 저장
        y_pred = [ 1 if x >= 0.5 else 0 for x in pred_probs ]
        get_clf_eval(y_test, y_pred, pred_probs)

        ### @ save : plot tree & plot importance feature 
        make_plot_tree(xgb_model, output_result_dir, outcome_name, rankdir=None)
        make_plot_tree(xgb_model, output_result_dir, outcome_name, rankdir='LR')
        make_plot_importance(xgb_model, output_result_dir, outcome_name)

        ### @ save : clf report & model estimation & confusion matrix & roc
        clf_report(y_test, y_pred, output_result_dir, outcome_name)
        model_performance_evaluation(y_test, y_pred, pred_probs, output_result_dir, outcome_name)
        confusion_matrix_figure(y_test, y_pred, output_result_dir, outcome_name)
        confusion_matrix_figure2(y_test, y_pred, output_result_dir, outcome_name)
        AUC, ACC = ROC_AUC(y_test, y_pred, output_result_dir, outcome_name)

        ### @ save : model json
        save_xgb_model_json(xgb_model, output_result_dir, outcome_name)
    
    except :
        traceback.print_exc()
        log.error(traceback.format_exc())

# %%
