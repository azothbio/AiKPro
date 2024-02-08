import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger, DataStructs, RDConfig
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm3D import Pharmacophore
from sklearn import preprocessing
from tqdm import tqdm
import pickle
import parmap
import math
import os
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
    df['3d'] = df[args.smiles].progress_apply(lambda x: smiles[x]['3d'] if x in smiles.keys() else 'NaN')
    df = df[df['3d'] != 'NaN'].reset_index(drop = True)
    df['pc'] = df[args.smiles].progress_apply(lambda x: smiles[x]['pc'] if x in smiles.keys() else 'NaN')
    df = df[df['pc'] != 'NaN'].reset_index(drop = True)
    df['mc'] = df[args.smiles].progress_apply(lambda x: smiles[x]['mc'] if x in smiles.keys() else 'NaN')
    df = df[df['mc'] != 'NaN'].reset_index(drop = True)
    df['mg'] = df[args.smiles].progress_apply(lambda x: smiles[x]['mg'] if x in smiles.keys() else 'NaN')
    df = df[df['mg'] != 'NaN'].reset_index(drop = True)
    df['en'] = df[args.name].progress_apply(lambda x: entry[x]['en'] if x in entry.keys() else 'NaN')
    df = df[df['en'] != 'NaN'].reset_index(drop = True)
    
    xki = np.vstack(df['en'])
    xpc = np.vstack(df['pc'])
    xmc = np.vstack(df['mc'])
    xmg = np.vstack(df['mg'])
    x3d = np.vstack(df['3d'])
    if not test:
        ytg = np.vstack(df[args.target])
        pc_factor = np.load(args.data+'_pc_factor.npy')
    else:
        pc_factor = np.load(args.pc_norm)
    
    print('- Data:', df.shape)
    print('- Name:', xki.shape)
    print('- PC', xpc.shape)
    print('- MC', xmc.shape)
    print('- MG', xmg.shape)
    print('- 3D', x3d.shape)
    if not test:
        print('- Y', ytg.shape)
    print('- PC_factor.npy:', pc_factor.shape)
    if not test:
        return df, xki, xpc, xmc, xmg, x3d, ytg, pc_factor
    else:
        return df, xki, xpc, xmc, xmg, x3d, pc_factor

def make_datasets(args, df, xki, xpc, xmc, xmg, x3d, ytg, pc_factor):
    train_df, test_df = train_test_split(df, 
                                         test_size=args.test_ratio, 
                                         random_state=args.seed)
    train_xki = xki[train_df.index]
    train_xpc = xpc[train_df.index]
    train_xpc = train_xpc/pc_factor
    train_xmc = xmc[train_df.index]
    train_xmg = xmg[train_df.index]
    train_x3d = x3d[train_df.index]
    train_ytg = ytg[train_df.index]
    train_data = [train_xki, train_xpc, train_xmc, train_xmg, train_x3d, train_ytg]
    
    test_xki = xki[test_df.index]
    test_xpc = xpc[test_df.index]
    test_xpc = test_xpc/pc_factor
    test_xmc = xmc[test_df.index]
    test_xmg = xmg[test_df.index]
    test_x3d = x3d[test_df.index]
    test_ytg = ytg[test_df.index]
    test_data = [test_xki, test_xpc, test_xmc, test_xmg, test_x3d, test_ytg]
    
    print('- Train:', train_df.shape[0])
    print('- Test:', test_df.shape[0])
    print()
    return args, train_data, test_data

def make_kfold_datasets(args, kfold, k, df, xki, xpc, xmc, xmg, x3d, ytg, pc_factor):
    for i, [train_idx, test_idx] in enumerate(kfold.split(df)):
        if k == i:
            train_xki = xki[train_idx]
            train_xpc= xpc[train_idx]
            train_xpc = train_xpc/pc_factor
            train_xmc = xmc[train_idx]
            train_xmg = xmg[train_idx]
            train_x3d = x3d[train_idx]
            train_ytg = ytg[train_idx]
            train_data = [train_xki, train_xpc, train_xmc, train_xmg, train_x3d, train_ytg]
            
            test_xki = xki[test_idx]
            test_xpc= xpc[test_idx]
            test_xpc = test_xpc/pc_factor
            test_xmc = xmc[test_idx]
            test_xmg = xmg[test_idx]
            test_x3d = x3d[test_idx]
            test_ytg = ytg[test_idx]
            test_data = [test_xki, test_xpc, test_xmc, test_xmg, test_x3d, test_ytg]
    if k == 0:
        print('- Train:', len(train_idx))
        print('- Test:', len(test_idx))
        print()
    return args, train_data, test_data
