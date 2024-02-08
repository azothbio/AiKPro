import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn import preprocessing
from tqdm import tqdm
import pickle
import parmap
import math
import warnings
warnings.filterwarnings(action='ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def load_datasets(args, test=False):
    print('[ Loading data ]')
    df = pd.read_csv(args.df)
    
    with open(file=args.data+'_entry.pickle', mode='rb') as f:
        entry = pickle.load(f)
    with open(file=args.data+'_smiles.pickle', mode='rb') as f:
        smiles = pickle.load(f)
    
    tqdm.pandas()
    df['pc'] = df[args.smiles].progress_apply(lambda x: smiles[x]['pc'] if x in smiles.keys() else 'NaN')
    df = df[df['pc'] != 'NaN'].reset_index(drop = True)
    df['mc'] = df[args.smiles].progress_apply(lambda x: smiles[x]['mc'] if x in smiles.keys() else 'NaN')
    df = df[df['mc'] != 'NaN'].reset_index(drop = True)
    df['mg'] = df[args.smiles].progress_apply(lambda x: smiles[x]['mg'] if x in smiles.keys() else 'NaN')
    df = df[df['mg'] != 'NaN'].reset_index(drop = True)
    df['msa'] = df[args.name].progress_apply(lambda x: entry[x]['msa'] if x in entry.keys() else 'NaN')
    df = df[df['msa'] != 'NaN'].reset_index(drop = True)
    
    xsa = np.array([x for x in df['msa']])
    xpc = np.vstack(df['pc'])
    xmc = np.vstack(df['mc'])
    xmg = np.vstack(df['mg'])
    if not test:
        ytg = np.vstack(df[args.target])
        pc_factor = np.load(args.data+'_pc_factor.npy')
    else:
        pc_factor = np.load(args.pc_norm)
    
    print('- Data:', df.shape)
    print('- MSA:', xsa.shape)
    print('- PC', xpc.shape)
    print('- MC', xmc.shape)
    print('- MG', xmg.shape)
    if not test:
        print('- Y', ytg.shape)
    print('- PC_factor.npy:', pc_factor.shape)
    if not test:
        return df, xsa, xpc, xmc, xmg, ytg, pc_factor
    else:
        return df, xsa, xpc, xmc, xmg, pc_factor

def make_datasets(args, df, xsa, xpc, xmc, xmg, ytg, pc_factor):
    train_df, test_df = train_test_split(df, 
                                         test_size=args.test_ratio, 
                                         random_state=args.seed)
    train_xsa = xsa[train_df.index]
    train_xpc = xpc[train_df.index]
    train_xpc = train_xpc/pc_factor
    train_xmc = xmc[train_df.index]
    train_xmg = xmg[train_df.index]
    train_ytg = ytg[train_df.index]
    train_data = [train_xsa, train_xpc, train_xmc, train_xmg, train_ytg]
    
    test_xsa = xsa[test_df.index]
    test_xpc = xpc[test_df.index]
    test_xpc = test_xpc/pc_factor
    test_xmc = xmc[test_df.index]
    test_xmg = xmg[test_df.index]
    test_ytg = ytg[test_df.index]
    test_data = [test_xsa, test_xpc, test_xmc, test_xmg, test_ytg]
    
    print('- Train:', train_df.shape[0])
    print('- Test:', test_df.shape[0])
    print()
    return args, train_data, test_data

def make_kfold_datasets(args, kfold, k, df, xsa, xpc, xmc, xmg, ytg, pc_factor):
    for i, [train_idx, test_idx] in enumerate(kfold.split(df)):
        if k == i:
            train_xpc= xpc[train_idx]
            train_xsa= xsa[train_idx]
            train_xpc = train_xpc/pc_factor
            train_xmc = xmc[train_idx]
            train_xmg = xmg[train_idx]
            train_ytg = ytg[train_idx]
            train_data = [train_xsa, train_xpc, train_xmc, train_xmg, train_ytg]
            
            test_xsa = xsa[test_idx]
            test_xpc= xpc[test_idx]
            test_xpc = test_xpc/pc_factor
            test_xmc = xmc[test_idx]
            test_xmg = xmg[test_idx]
            test_ytg = ytg[test_idx]
            test_data = [test_xsa, test_xpc, test_xmc, test_xmg, test_ytg]
    if k == 0:
        print('- Train:', len(train_idx))
        print('- Test:', len(test_idx))
        print()
    return args, train_data, test_data
