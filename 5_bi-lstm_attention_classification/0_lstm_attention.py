#!/usr/bin/env python
# coding: utf-8

'''
lstm attention 
'''
# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib
sys.path.append("..")

import traceback
from tqdm import tqdm
from _utils.Auto_lstm_attention import *
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

# In[ ]:
# for linux
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# In[ ]:
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        # In[ ]:
        log.debug("{}".format(outcome_name))
        ps_data_dir = pathlib.Path('{}/data/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))
        output_dir = pathlib.Path('{}/result/{}/lstm_attention/{}/'.format(parent_dir, current_date, outcome_name))
        pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)

        def split_x_y_data(df, OBP) :
            import numpy as np
            import pandas as pd

            y_data = df['label'].T.reset_index(drop=True) #df['label'].T.drop_duplicates().T.reset_index(drop=True)
            y_data = np.array(y_data)
            y_data = y_data[0:len(y_data):OBP].reshape(-1, 1).astype(int)
            #print(len(y_data), file=_logfile_)

            x_df = df.drop('label', axis=1)

            # 2-d data to 3-d data
            timestamp = OBP 
            x_data = np.array(x_df)
            x_data = x_data.reshape(-1, timestamp, x_data.shape[1]) # -1(sample), timestamp, column
            #x_data.shape, y_data.shape

            # get Column data
            new_col = x_df.columns
            print(x_data.shape, y_data.shape, len(new_col))
            return x_data, y_data, new_col

        c = Auto_lstm_attention()
                
        concat_df = pd.read_csv('{}/{}.txt'.format(ps_data_dir, outcome_name), index_col=False)
        
        # ##### Case 1 : Split by person_id #####
        # id_data = concat_df[['person_id', 'label']].drop_duplicates().reset_index(drop=True)
        # x_id_data = np.array(id_data['person_id'])
        # y_id_data = np.array(id_data['label'])
        
        # x_id_train, x_id_test, y_id_train, y_id_test = train_test_split(x_id_data, y_id_data, test_size=0.3, random_state=1, stratify=y_id_data) 
        
        # train_df = concat_df[concat_df['person_id'].isin(x_id_train)].reset_index(drop=True)
        # test_df = concat_df[concat_df['person_id'].isin(x_id_test)].reset_index(drop=True)
        
        # train_df.to_csv('{}/{}_train.txt'.format(output_dir, outcome_name), index=False)
        # test_df.to_csv('{}/{}_test.txt'.format(output_dir, outcome_name), index=False)
        
        # concat_df = concat_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
        # train_df = train_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
        # test_df = test_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
        
        # x_data, y_data, new_col = split_x_y_data(concat_df, OBP=28)
        # x_train, y_train, new_col = split_x_y_data(train_df, OBP=28)
        # x_test, y_test, new_col = split_x_y_data(test_df, OBP=28)
        
        #### Case 2 : Split ignore person_id #####
        concat_df = concat_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
    
        x_data, y_data, new_col = split_x_y_data(concat_df, OBP=7)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1, stratify=y_data) 
        
        c = Auto_lstm_attention()
        model = c.LSTM_attention_building(x_train, x_test, y_train, y_test)
        weights = c.class_balance_weight(output_dir, outcome_name, y_train)
        h, y_hat = c.early_stopping_prediction()
        c.classification_report(output_dir, outcome_name)
        c.model_performance_evaluation(output_dir, outcome_name)
        c.confusion_matrix_figure(output_dir, outcome_name)
        c.confusion_matrix_figure2(output_dir, outcome_name)
        AUC, ACC = c.ROC_AUC(output_dir, outcome_name)
        c.loss(output_dir, outcome_name)
        
        if_ = c.attention_heatmap(new_col, output_dir, outcome_name)
        accuracy = c.k_fold_cross_validation(x_data, y_data, output_dir, outcome_name)
        
        # # (['AUC','ACC','import_f', 'k_fold'])
        # model.save('{}/{}.h5'.format(output_dir, outcome_name))

        out = open('{}/output.txt'.format(output_dir),'a')
        
        out.write(str(outcome_name))
        out.write('///' )
        out.write(str(AUC ))
        out.write('///' )
        out.write(str(ACC ))
        out.write('///' )
        out.write(str(str(if_) ))
        out.write('///' )
        out.write(str(accuracy))
        out.write('\n')
        
        out.close()

    except :
        traceback.print_exc()
        log.error(traceback.format_exc())
        

# %%
