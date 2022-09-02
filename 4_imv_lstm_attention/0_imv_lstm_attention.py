# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib
sys.path.append("..")

import traceback
from tqdm import tqdm
import textwrap
from _utils.Auto_lstm_attention import *
from _utils.model_estimation import *
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imv_lstm_model import IMVFullLSTM
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# In[ ]:
# for linux
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# In[ ]:
def get_time_stamp(df):
    return int(len(df)/len(df.unique_id.unique()))

def split_x_y_data(df, n_timestamp):
    import numpy as np
    import pandas as pd

    y_data = df['label'].T.reset_index(drop=True) #df['label'].T.drop_duplicates().T.reset_index(drop=True)
    y_data = np.array(y_data)
    y_data = y_data[0:len(y_data):n_timestamp].reshape(-1, 1).astype(int)
    #print(len(y_data), file=_logfile_)

    X_df = df.drop('label', axis=1)

    # 2-d data to 3-d data
    X_data = np.array(X_df)
    X_data = X_data.reshape(-1, n_timestamp, X_data.shape[1]) # -1(sample), timestamp, column
    #X_data.shape, y_data.shape

    # get Column data
    new_col = X_df.columns
    print(X_data.shape, y_data.shape, len(new_col))
    return X_data, y_data, new_col

# In[ ]:
def runTask(outcome_name):
    ps_data_dir = pathlib.Path('{}/data/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))
    output_data_dir = pathlib.Path('{}/data/{}/imv_lstm_attention/{}/'.format(parent_dir, current_date, outcome_name))
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    output_result_dir = pathlib.Path('{}/result/{}/imv_lstm_attention/{}/'.format(parent_dir, current_date, outcome_name))
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

    concat_df = pd.read_csv('{}/{}_4w.txt'.format(ps_data_dir, outcome_name), index_col=False)
    if concat_df.empty:
            print(outcome_name, " is empty")
            return

    # ##### Case 1 : Split by person_id #####
    # id_data = concat_df[['person_id', 'label']].drop_duplicates().reset_index(drop=True)
    # x_id_data = np.array(id_data['person_id'])
    # y_id_data = np.array(id_data['label'])
    
    # x_id_train, x_id_test, y_id_train, y_id_test = train_test_split(x_id_data, y_id_data, test_size=0.3, random_state=1, stratify=y_id_data) 
    
    # train_df = concat_df[concat_df['person_id'].isin(x_id_train)].reset_index(drop=True)
    # test_df = concat_df[concat_df['person_id'].isin(x_id_test)].reset_index(drop=True)
    
    # train_df.to_csv('{}/{}_train.txt'.format(output_result_dir, outcome_name), index=False)
    # test_df.to_csv('{}/{}_test.txt'.format(output_result_dir, outcome_name), index=False)
    
    # concat_df = concat_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
    # train_df = train_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
    # test_df = test_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
    
    # x_data, y_data, new_col = split_x_y_data(concat_df, OBP=28)
    # x_train, y_train, new_col = split_x_y_data(train_df, OBP=28)
    # x_test, y_test, new_col = split_x_y_data(test_df, OBP=28)
    
    n_timestamp = get_time_stamp(concat_df) 
    #### Case 2 : Split ignore person_id #####
    concat_df = concat_df.drop(['person_id', 'unique_id', 'cohort_start_date', 'concept_date', 'first_abnormal_date'], axis=1)
    
    X_data, y_data, cols = split_x_y_data(concat_df, n_timestamp)
    cols = [textwrap.shorten(col, width=50, placeholder="...") for col in cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=1, stratify=y_data) 

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    train_bound = int(0.8*(len(X_train)))

    X_val = X_train[train_bound:]
    X_train = X_train[:train_bound]
    y_val = y_train[train_bound:]
    y_train = y_train[:train_bound]
    depth = n_timestamp
    
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)

    X_train_t = torch.Tensor(X_train)
    X_val_t = torch.Tensor(X_val)
    X_test_t = torch.Tensor(X_test)
    y_train_t = torch.Tensor(y_train)
    y_val_t = torch.Tensor(y_val)
    y_test_t = torch.Tensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    for x, y in train_loader:
        print(y.shape)
        break

    model = IMVFullLSTM(X_train_t.shape[2], 1, 128)
    opt = torch.optim.Adam(model.parameters(), lr=params['learningrate'])
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
    
    epochs = params["epochs"]
    patience = params["patience"]
    min_val_loss = params["min_val_loss"]
    loss = nn.MSELoss()
    counter = 0
    for i in range(epochs):
        mse_train = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x
            batch_y = batch_y
            opt.zero_grad()
            y_pred, alphas, betas = model(batch_x)
            y_pred = y_pred.squeeze(1)
            l = loss(y_pred, batch_y)
            l.backward()
            mse_train += l.item()*batch_x.shape[0]
            opt.step()
        epoch_scheduler.step()
        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x
                batch_y = batch_y
                output, alphas, betas = model(batch_x)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += loss(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        
        if min_val_loss > mse_val**0.5:
            min_val_loss = mse_val**0.5
            print("Saving...")
            torch.save(model.state_dict(), "{}/{}_model_state_dict.pt".format(output_data_dir, outcome_name))
            counter = 0
        else: 
            counter += 1
        
        if counter == patience:
            break
        print("Iter: ", i, "train: ", (mse_train/len(X_train_t))**0.5, "val: ", (mse_val/len(X_val_t))**0.5)
        if(i % 10 == 0):
            preds = preds*(y_train_max - y_train_min) + y_train_min
            true = true*(y_train_max - y_train_min) + y_train_min
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            print("lr: ", opt.param_groups[0]["lr"])
            print("mse: ", mse, "mae: ", mae)
            plt.figure(figsize=(20, 10))
            plt.plot(preds)
            plt.plot(true)
            plt.show()

    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        alphas = []
        betas = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x
            batch_y = batch_y
            output, a, b = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            alphas.append(a.detach().cpu().numpy())
            betas.append(b.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item()*batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    preds = preds*(y_train_max - y_train_min) + y_train_min
    true = true*(y_train_max - y_train_min) + y_train_min
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    plt.figure(figsize=(20, 10))
    plt.plot(preds)
    plt.plot(true)
    plt.show()
    alphas = np.concatenate(alphas)
    betas = np.concatenate(betas)
    alphas = alphas.mean(axis=0)
    betas = betas.mean(axis=0)
    alphas = alphas[..., 0]
    betas = betas[..., 0]
    alphas = alphas.transpose(1, 0)

    figure_data = {}
    figure_data['alphas'] = alphas
    figure_data['betas'] = betas
    figure_data['cols'] = cols
    with open(output_result_dir.joinpath('figure_data.pickle'), 'wb') as f:
        import pickle
        pickle.dump(figure_data, f, pickle.HIGHEST_PROTOCOL)

    fig, ax = plt.subplots(figsize=(40, 30))
    im = ax.imshow(alphas)
    ax.set_xticks(np.arange(X_train_t.shape[1]))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(["t-"+str(i) for i in np.arange(X_train_t.shape[1], 0, -1)])
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(X_train_t.shape[1]):
            text = ax.text(j, i, round(alphas[i, j], 3),
                        ha="center", va="center", color="w")
    ax.set_title("Importance of features and timesteps")
    #fig.tight_layout()

    plt.savefig('{}/{}_heatmap_.png'.format(output_result_dir, outcome_name), format='png',
                        dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 15))
    plt.title("Feature importance")
    plt.barh(cols, betas)
    plt.gca().invert_yaxis()
    # plt.xticks(ticks=range(len(cols)), labels=list(cols), rotation=90)

    plt.savefig('{}/{}_Feature_importance_.png'.format(output_result_dir, outcome_name), format='png',
                        dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    plt.show()
    
    y_true = true
    y_pred_proba = preds
    y_pred = np.rint(preds)

    confusion_matrix_figure2(y_true, y_pred, output_result_dir, outcome_name)
    ROC_AUC(y_pred_proba, y_true, output_result_dir, outcome_name)
    model_performance_evaluation(y_true, y_pred, y_pred_proba, output_result_dir, outcome_name)

# In[ ]:
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        # In[ ]:
        runTask(outcome_name)
    except :
        traceback.print_exc()
        log.error(traceback.format_exc())

# %%
