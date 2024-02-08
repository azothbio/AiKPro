import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import dataset
import experiment

start = time.time()

parser = argparse.ArgumentParser(description='KPro')
parser.add_argument('--df', type=str, default='./Data/kpro_train.csv', help='input path and file name')
parser.add_argument('--data', type=str, default='./Data/KPro_Train', help='input path and file name')
parser.add_argument('--smiles', type=str, default='SMILES', help='smiles column name')
parser.add_argument('--name', type=str, default='Name', help='kinase entry name column name')
parser.add_argument('--kinase', type=str, default='MSA', help='kinase entry column name')
parser.add_argument('--target', '-y', type=str, default='Value', help='target column name')
parser.add_argument('--seed', type=int, default=2022, help='random state')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test set ratio')
parser.add_argument('--gpu', type=str, default='0', help='gpu name')
parser.add_argument('--log', type=str, default='./log/KPro_Model/', help='log file path')

parser.add_argument('--sa_conv_1dfilters', type=int, default=32, help='number of filter size for 1D convolution layers')
parser.add_argument('--sa_conv_1dkernel', type=int, default=3, help='number of kernel size for 1D convolution layers')
parser.add_argument('--sa_conv_1dstrides', type=int, default=1, help='number of strides size for 1D convolution layers')
parser.add_argument('--sa_pool1d', type=int, default=2, help='number of pool size for 1D convolution layers')

parser.add_argument('--conv_1dfilters', type=int, default=16, help='number of filter size for 1D convolution layers')
parser.add_argument('--conv_1dkernel', type=int, default=5, help='number of kernel size for 1D convolution layers')
parser.add_argument('--conv_1dstrides', type=int, default=1, help='number of strides size for 1D convolution layers')
parser.add_argument('--pool1d', type=int, default=2, help='number of pool size for 1D convolution layers')

parser.add_argument('--sa_n_conv1d', type=int, default=2, help='number of dense layers for kinase')
parser.add_argument('--pc_n_conv1d', type=int, default=1, help='number of dense layers for physicochemical properties')
parser.add_argument('--mc_n_conv1d', type=int, default=1, help='number of 1D convolution layers for MACCS fingerprints')
parser.add_argument('--mg_n_conv1d', type=int, default=3, help='number of 1D convolution layers for Morgan fingerprints')
parser.add_argument('--td_n_conv1d', type=int, default=3, help='number of 1D convolution layers for 3D fingerprints')
parser.add_argument('--ma_n_conv1d', type=int, default=1, help='number of 1D convolution layers for molecule concat dense layers')

parser.add_argument('--mhatt_num_heads', type=int, default=4, help='number of heads for multi-head attention')
parser.add_argument('--mhatt_key_dim', type=int, default=16, help='number of key dimensions for multi-head attention')

parser.add_argument('--sa_n_attention', type=int, default=4, help='number of multi-head attention layers for ki')
parser.add_argument('--pc_n_attention', type=int, default=4, help='number of multi-head attention layers for pc')
parser.add_argument('--mc_n_attention', type=int, default=4, help='number of multi-head attention layers for mc')
parser.add_argument('--mg_n_attention', type=int, default=4, help='number of multi-head attention layers for mg')
parser.add_argument('--td_n_attention', type=int, default=4, help='number of multi-head attention layers for 3d')
parser.add_argument('--ma_n_attention', type=int, default=2, help='number of multi-head attention layers for ma')
parser.add_argument('--cat_n_attention', type=int, default=2, help='number of multi-head attention layers for x')

parser.add_argument('--activation', type=str, default='relu', help='activation type')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')

parser.add_argument('--epochs', type=int, default=3000, help='epochs')
parser.add_argument('--batch', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--decay_steps', type=float, default=10000, help='decay steps for learning rate')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate for learning rate')
parser.add_argument('--patience', type=float, default=100, help='patience for early stopping')
parser.add_argument('--kfold', type=int, default=0, help='kfold validation')
parser.add_argument('--exp_name', type=str, default='KPro', help='experiment name')
parser.add_argument('--model_name', type=str, default='./Model/KPro_Total_Model', help='model name')
parser.add_argument('--result', type=str, default='./Result/KPro_Total_Result.json', help='output json file name')
args = parser.parse_args()

print('[ Hyperparameters ]')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

for arg in str(args)[10:-1].split(', '):
    print(f"- {': '.join(arg.split('='))}")
print()

if args.kfold == 0:
    df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor = dataset.load_datasets(args)
    args, train_data, test_data = dataset.make_datasets(args, df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor)
    dict_result = dict()
    args, kpro = experiment.experiment(args, train_data, test_data)
    args.time = time.time() - start
    dict_result[args.exp_name] = vars(args)
    result_df = pd.DataFrame(dict_result).transpose()
    result_df.to_json(args.result, orient='table')
    kpro.save_weights(args.model_name)
else:
    df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor = dataset.load_datasets(args)
    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for k in range(args.kfold):
        args, train_data, test_data = dataset.make_kfold_datasets(args, kfold, k, 
                                                                  df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor)
        k += 1
        if k > 1:
            print(f'- K-Fold Validation: {k}/{str(int(args.kfold))}')
            print()
            dict_result = dict()
            args, kpro = experiment.experiment(args, train_data, test_data)
            args.time = time.time() - start
            dict_result[f'{args.exp_name}_{k}'] = vars(args)
            result_df = pd.DataFrame(dict_result).transpose()
            result_df.to_json(f'{str(args.result)[:-5]}_{k}.json', orient='table')
            kpro.save_weights(f'{args.model_name}_{k}')
            print()

