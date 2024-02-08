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


np.random.seed(2022)

def __init_Fac():
    fName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    fFac = ChemicalFeatures.BuildFeatureFactory(fName)
    sFac = SigFactory(fFac, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
    sFac.SetBins([(0,3), (3,6), (6,9)])
    sFac.Init()
    return fFac, sFac

_fFac, _sFac = __init_Fac()
_gn_prms = {'numConfs': 16, 
            'maxAttempts': 1, 
            'maxIters': 200, 
            'maxThreads': 32}

def set_gn_prms(numConfs=16, maxAttempts=1, maxIters=200, maxThreads=32):
    global _gn_prms
    _gn_prms['numConfs'] = numConfs
    _gn_prms['maxAttempts'] = maxAttempts
    _gn_prms['maxIters'] = maxIters
    _gn_prms['maxThreads'] = maxThreads

numConfs=16
maxAttempts=10
maxIters=200
maxThreads=64
set_gn_prms(numConfs=numConfs, maxAttempts=maxAttempts, maxIters=maxIters, maxThreads=maxThreads)

def calxki(inputs):
    k, names, kinase_to_int = inputs
    x = np.zeros(len(names), dtype = np.int8)
    x[kinase_to_int[k]] =1
    return x

seq = ['L', 'D', 'P', 'K', 'Y', 'Q', 'E', 'W', 'G', 'I', 'S', 'N', 'C', 'H', 'F', 'T', 'A', 'R', 'V', 'M', '-']
seq_to_int = dict((s, i) for i, s in enumerate(seq))
def calxsa(msa):
    xsa = np.array(tf.keras.utils.to_categorical([seq_to_int[s] for s in msa], num_classes=21))
    return xsa

def calxpc(csmi):
    try:
        mol = Chem.MolFromSmiles(csmi)
        lx = []
        for n, fn_pc in Descriptors.descList:
            lx.append(fn_pc(mol))
        nan = 0
        for x in lx:
            if str(x) == 'nan':
                nan = 1
            if str(x) == 'inf':
                nan = 1
        if nan == 0:
            return (csmi, lx)
        else:
            return (csmi, None)
    except:
        return (csmi, None)

def calxmc(csmi):
    try:
        mol = Chem.MolFromSmiles(csmi)
        mfp = MACCSkeys.GenMACCSKeys(mol)
        lx = np.zeros(167, dtype=np.int8)
        lx[mfp.GetOnBits()] = 1
        return (csmi, lx)
    except:
        return (csmi, None)
        
def calxmg(csmi):
    lbits = [256, 512, 1024]
    try:
        mol = Chem.MolFromSmiles(csmi)
        lx = []
        for i,b in enumerate(lbits):
            m1 = AllChem.GetMorganFingerprintAsBitVect(mol, i+2, b, useFeatures=False)
            mx = np.zeros(b, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(m1, mx)
            lx = lx + mx.tolist()
        return (csmi, lx)
    except:
        return (csmi, None)

def calx3d(csmi):
    try:
        mol = Chem.MolFromSmiles(csmi)
        mh = Chem.AddHs(mol)
        m_confs = AllChem.EmbedMultipleConfs(mh, 
                                             numConfs=_gn_prms['numConfs'], 
                                             maxAttempts=_gn_prms['maxAttempts'], 
                                             useRandomCoords=True)
        ret = AllChem.UFFOptimizeMoleculeConfs(mh, maxIters=_gn_prms['maxIters'])
        lx = []
        m = Chem.RemoveHs(mh)
        for i in m_confs:
            fp = Generate.Gen2DFingerprint(m, _sFac, dMat=Chem.Get3DDistanceMatrix(m, confId=i))
            xfp = np.zeros(fp.GetNumBits(), dtype=np.int8)
            xfp[fp.GetOnBits()] = 1
            lx.append(xfp)
        x3d = np.array(lx, dtype=np.int8).sum(axis=0)/16
        return (csmi, x3d)
    except:
        return (csmi, None)

def calc_desc(l_csmi, msas, kinase, names, kinase_to_int, target=None, cpus=10):
    # Xki
    l_kinase = parmap.map(calxki, [[k, names, kinase_to_int] for k in kinase], 
                          pm_pbar=True, pm_processes=cpus)
    # Xsa
    l_msa = parmap.map(calxsa, [msa.upper() for msa in msas], 
                       pm_pbar=True, pm_processes=cpus)
    # Xpc
    l_zip = parmap.map(calxpc, l_csmi, pm_pbar=True, pm_processes=cpus)
    l_csmi_pc, l_xpc = zip(*l_zip)
    d_c2i_pc = {c:i for i, c in enumerate(l_csmi_pc)}
    # Xmc
    l_zip = parmap.map(calxmc, l_csmi, pm_pbar=True, pm_processes=cpus)
    l_csmi_mc, l_xmc = zip(*l_zip)
    d_c2i_mc = {c:i for i, c in enumerate(l_csmi_mc)}
    # Xmg
    l_zip = parmap.map(calxmg, l_csmi, pm_pbar=True, pm_processes=cpus)
    l_csmi_mg, l_xmg = zip(*l_zip)
    d_c2i_mg = {c:i for i, c in enumerate(l_csmi_mg)}
    # X3d
    l_zip = parmap.map(calx3d, l_csmi, pm_pbar=True, pm_processes=cpus)
    l_csmi_3d, l_x3d = zip(*l_zip)
    d_c2i_3d = {c:i for i, c in enumerate(l_csmi_3d)}
    
    entry = {}
    smiles = {}
    for i, k in enumerate(kinase):
        entry[k] = {'en':np.array(l_kinase[i], dtype=np.float32), 
                    'msa':np.array(l_msa[i], dtype=np.float32)}
    
    for i, c in enumerate(l_csmi):
        if l_xpc[d_c2i_pc[c]] is None:
            continue
        if l_xmc[d_c2i_mc[c]] is None:
            continue
        if l_xmg[d_c2i_mg[c]] is None:
            continue
        if l_x3d[d_c2i_3d[c]] is None:
            continue
        smiles[c] = {'pc':np.array(l_xpc[d_c2i_pc[c]], dtype=np.float64), 
                     'mc':np.array(l_xmc[d_c2i_mc[c]], dtype=np.float32), 
                     'mg':np.array(l_xmg[d_c2i_mg[c]], dtype=np.float32), 
                     '3d':np.array(l_x3d[d_c2i_3d[c]], dtype=np.float32)}
    
    return entry, smiles

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
    df['msa'] = df[args.name].progress_apply(lambda x: entry[x]['msa'] if x in entry.keys() else 'NaN')
    df = df[df['msa'] != 'NaN'].reset_index(drop = True)
    
    xsa = np.array([x for x in df['msa']])
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
    print('- MSA:', xsa.shape)
    print('- PC', xpc.shape)
    print('- MC', xmc.shape)
    print('- MG', xmg.shape)
    print('- 3D', x3d.shape)
    if not test:
        print('- Y', ytg.shape)
    print('- PC_factor.npy:', pc_factor.shape)
    if not test:
        return df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor
    else:
        return df, xsa, xpc, xmc, xmg, x3d, pc_factor

def make_datasets(args, df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor):
    train_df, test_df = train_test_split(df, 
                                         test_size=args.test_ratio, 
                                         random_state=args.seed)
    train_xsa = xsa[train_df.index]
    train_xpc = xpc[train_df.index]
    train_xpc = train_xpc/pc_factor
    train_xmc = xmc[train_df.index]
    train_xmg = xmg[train_df.index]
    train_x3d = x3d[train_df.index]
    train_ytg = ytg[train_df.index]
    train_data = [train_xsa, train_xpc, train_xmc, train_xmg, train_x3d, train_ytg]
    
    test_xsa = xsa[test_df.index]
    test_xpc = xpc[test_df.index]
    test_xpc = test_xpc/pc_factor
    test_xmc = xmc[test_df.index]
    test_xmg = xmg[test_df.index]
    test_x3d = x3d[test_df.index]
    test_ytg = ytg[test_df.index]
    test_data = [test_xsa, test_xpc, test_xmc, test_xmg, test_x3d, test_ytg]
    
    print('- Train:', train_df.shape[0])
    print('- Test:', test_df.shape[0])
    print()
    return args, train_data, test_data

def make_kfold_datasets(args, kfold, k, df, xsa, xpc, xmc, xmg, x3d, ytg, pc_factor):
    for i, [train_idx, test_idx] in enumerate(kfold.split(df)):
        if k == i:
            train_xpc= xpc[train_idx]
            train_xsa= xsa[train_idx]
            train_xpc = train_xpc/pc_factor
            train_xmc = xmc[train_idx]
            train_xmg = xmg[train_idx]
            train_x3d = x3d[train_idx]
            train_ytg = ytg[train_idx]
            train_data = [train_xsa, train_xpc, train_xmc, train_xmg, train_x3d, train_ytg]
            
            test_xsa = xsa[test_idx]
            test_xpc= xpc[test_idx]
            test_xpc = test_xpc/pc_factor
            test_xmc = xmc[test_idx]
            test_xmg = xmg[test_idx]
            test_x3d = x3d[test_idx]
            test_ytg = ytg[test_idx]
            test_data = [test_xsa, test_xpc, test_xmc, test_xmg, test_x3d, test_ytg]
    if k == 0:
        print('- Train:', len(train_idx))
        print('- Test:', len(test_idx))
        print()
    return args, train_data, test_data
