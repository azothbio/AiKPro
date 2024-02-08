import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import argparse
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import dataset


parser = argparse.ArgumentParser(description='KPro')
parser.add_argument('--file', type=str, default='../Data/kpro_data_train.csv', help='input csv file name')
parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name')
parser.add_argument('--name', type=str, default='Name', help='kinase Entry name') #
parser.add_argument('--msa', type=str, default='MSA', help='kinase MSA name')
parser.add_argument('--seed', type=int, default=2022, help='random state')
parser.add_argument('--cpus', type=int, default=50, help='number of cpus')
parser.add_argument('--data', type=str, default='./Data/KPro_Train', help='output file path and name')
parser.add_argument('--pc', action='store_false', help='save px norm factor')
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

print('[ Hyperparameters ]')
for arg in str(args)[10:-1].split(', '):
    print(f"- {': '.join(arg.split('='))}")
print()

df = pd.read_csv(args.file)
kinase = list(set(df[args.msa].tolist()))
names = list(set(df[args.name].tolist())) #
kinase_to_int = dict((k, i) for i, k in enumerate(names)) #

df['Converting'] = 0
print('[ Converting to descriptors ]')
print('- Shape of this dataset:', df.shape)
print('- Types of kinase:', len(kinase))
entry, smiles = dataset.calc_desc(df[args.smiles].unique().tolist(), #
                                  df[args.msa].unique().tolist(), 
                                  df[args.name].unique().tolist(), #
                                  names, #
                                  kinase_to_int, #
                                  cpus=args.cpus)

print()
print('[ Saving data ]')
with open(file=args.data+'_entry.pickle', mode='wb') as f:
    pickle.dump(entry, f)
with open(file=args.data+'_smiles.pickle', mode='wb') as f:
    pickle.dump(smiles, f)

if args.pc:
    pcs = []
    for k in smiles.keys():
        pcs.append(smiles[k]['pc'])
    pcs = np.array(pcs, dtype=np.float64)
    pc_factor = np.abs(pcs).max(axis=0)
    pc_factor = np.where(pc_factor != 0, pc_factor, 1)
    np.save(args.data+'_pc_factor.npy', pc_factor)
    print('Saved px norm factor!')

