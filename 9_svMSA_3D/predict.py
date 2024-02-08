import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import dataset
import model
import experiment


parser = argparse.ArgumentParser(description='KPro')
args = parser.parse_args('')
parser.add_argument('--df', type=str, default='./Data/kpro_test.csv', help='input path and file name')
parser.add_argument('--data', type=str, default='./Data/KPro_Test', help='input path and file name')
parser.add_argument('--smiles', type=str, default='SMILES', help='smiles column name')
parser.add_argument('--name', type=str, default='Name', help='kinase entry name column name')
parser.add_argument('--msa', type=str, default='MSA', help='MSA column name')
parser.add_argument('--load_results', type=str, default='./Result/KPro_Result_1.json,./Result/KPro_Result_2.json,./Result/KPro_Result_3.json,./Result/KPro_Result_4.json,./Result/KPro_Total_5.json', help='output json file name')
parser.add_argument('--load_models', type=str, default='./Model/KPro_Model_1,./Model/KPro_Model_2,./Model/KPro_Model_3,./Model/KPro_Model_4,./Model/KPro_Model_5', help='output json file name')
parser.add_argument('--pc_norm', type=str, default='../Data/KPro_Train_pc_factor.npy', help='pc norm factor')
parser.add_argument('--seed', type=int, default=2022, help='random state')
parser.add_argument('--cpus', type=int, default=100, help='number of cpus')
parser.add_argument('--gpu', type=str, default='0,1', help='gpu name')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--exp_name', type=str, default='Pred', help='experiment name')
parser.add_argument('--result', type=str, default='./Predict/Pred_Total_Result', help='output file name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

print('[ Hyperparameters ]')
for arg in str(args)[10:-1].split(', '):
    print(f"- {': '.join(arg.split('='))}")

load_results = args.load_results.split(',')
load_models = args.load_models.split(',')

if len(load_results) != len(load_models):
    raise ValueError('results and models loading error.')

print(f"- number_of_models:", len(load_models))
total_pred_list = []
k = 1
print()

df, test_xki, test_xpc, test_xmc, test_xmg, test_x3d, maxpc = dataset.load_datasets(args, test=True)

print()
print('[ Prediction ]')
for load_result, model_name in zip(load_results, load_models):
    print(f'- Model {k}/{len(load_models)}')
    result_df = pd.read_json(load_result, orient='table')
    args.sa_conv_1dfilters = result_df['sa_conv_1dfilters'].iloc[0]
    args.sa_conv_1dkernel = result_df['sa_conv_1dkernel'].iloc[0]
    args.sa_conv_1dstrides = result_df['sa_conv_1dstrides'].iloc[0]
    args.sa_pool1d = result_df['sa_pool1d'].iloc[0]
    args.conv_1dfilters = result_df['conv_1dfilters'].iloc[0]
    args.conv_1dkernel = result_df['conv_1dkernel'].iloc[0]
    args.conv_1dstrides = result_df['conv_1dstrides'].iloc[0]
    args.pool1d = result_df['pool1d'].iloc[0]
    args.sa_n_conv1d = result_df['sa_n_conv1d'].iloc[0]
    args.pc_n_conv1d = result_df['pc_n_conv1d'].iloc[0]
    args.mc_n_conv1d = result_df['mc_n_conv1d'].iloc[0]
    args.mg_n_conv1d = result_df['mg_n_conv1d'].iloc[0]
    args.td_n_conv1d = result_df['td_n_conv1d'].iloc[0]
    args.ma_n_conv1d = result_df['ma_n_conv1d'].iloc[0]
    args.sa_n_attention = result_df['sa_n_attention'].iloc[0]
    args.pc_n_attention = result_df['pc_n_attention'].iloc[0]
    args.mc_n_attention = result_df['mc_n_attention'].iloc[0]
    args.mg_n_attention = result_df['mg_n_attention'].iloc[0]
    args.td_n_attention = result_df['td_n_attention'].iloc[0]
    args.ma_n_attention = result_df['ma_n_attention'].iloc[0]
    args.cat_n_attention = result_df['cat_n_attention'].iloc[0]
    args.mhatt_num_heads = result_df['mhatt_num_heads'].iloc[0]
    args.mhatt_key_dim = result_df['mhatt_key_dim'].iloc[0]
    args.activation = result_df['activation'].iloc[0]
    args.dropout = result_df['dropout'].iloc[0]
    test_xpc = test_xpc/maxpc
    
    kpro = model.KPro(args)
    kpro.load_weights(model_name)
    
    pred_list = kpro.predict([test_xki, test_xpc, test_xmc, test_xmg, test_x3d], batch_size=args.batch)
    args.pred_list = pred_list
    total_pred_list.append(pred_list)
    
    df[f'Predicted_{k}'] = pred_list
    df.to_csv(args.result+f'_{k}.csv', index=False)

    dict_result = dict()
    dict_result[args.exp_name] = vars(args)
    result_df = pd.DataFrame(dict_result).transpose()
    result_df.to_json(args.result+f'_{k}.json', orient='table')
    k += 1

pred_ens_df = pd.concat([pd.read_csv(f'{args.result}_1.csv').iloc[:,:-1]] + 
                        [pd.read_csv(f'{args.result}_{i}.csv')[[f'Predicted_{i}']] for i in range(1, len(load_models)+1)], 
                        axis=1)
pred_ens_df = pred_ens_df.dropna().reset_index(drop=True)
pred_ens_df['Predicted'] = pred_ens_df[[f'Predicted_{i}' for i in range(1, len(load_models)+1)]].mean(axis='columns')
pred_ens_df.to_csv(args.result+'_Pred_Means.csv', index=False)
print('Done!')
