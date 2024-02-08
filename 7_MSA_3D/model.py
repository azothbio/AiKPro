import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as kL
from tensorflow.keras import models as kM
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

prints = False

class KPro(Model):
    
    def __init__(self, args):
        super().__init__()
        
        self.sa_pre_conv1d_layers = []
        self.sa_pre_conv1d_layers.append(kL.Conv1D(filters=args.sa_conv_1dfilters, 
                                                   kernel_size=5, 
                                                   strides=2))
        self.sa_pre_conv1d_layers.append(kL.BatchNormalization())
        self.sa_pre_conv1d_layers.append(kL.ReLU())
        self.sa_pre_conv1d_layers.append(kL.Dropout(args.dropout))
        self.sa_pre_conv1d_layers.append(kL.MaxPool1D(pool_size=1))
        
        self.sa_conv1d_layers = []
        for i in range(args.sa_n_conv1d-1):
            self.sa_conv1d_layers.append(kL.Conv1D(filters=args.sa_conv_1dfilters, 
                                                   kernel_size=args.sa_conv_1dkernel, 
                                                   strides=args.sa_conv_1dstrides))
            self.sa_conv1d_layers.append(kL.BatchNormalization())
            self.sa_conv1d_layers.append(kL.ReLU())
            self.sa_conv1d_layers.append(kL.Dropout(args.dropout))
            self.sa_conv1d_layers.append(kL.MaxPool1D(pool_size=args.sa_pool1d))
            self.sa_conv1d_layers.append(kL.Conv1D(filters=args.sa_conv_1dfilters, 
                                                   kernel_size=args.sa_conv_1dkernel))
            self.sa_conv1d_layers.append(kL.BatchNormalization())
            
        self.pc_conv1d_layers = []
        for i in range(args.pc_n_conv1d):
            self.pc_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.pc_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.pc_conv1d_layers.append(kL.Dropout(args.dropout))
            self.pc_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.mc_conv1d_layers = []
        for i in range(args.mc_n_conv1d):
            self.mc_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.mc_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.mc_conv1d_layers.append(kL.Dropout(args.dropout))
            self.mc_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.mg_conv1d_layers = []
        for i in range(args.mg_n_conv1d):
            self.mg_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.mg_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.mg_conv1d_layers.append(kL.Dropout(args.dropout))
            self.mg_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.td_conv1d_layers = []
        for i in range(args.td_n_conv1d):
            self.td_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.td_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.td_conv1d_layers.append(kL.Dropout(args.dropout))
            self.td_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.ma_conv1d_layers = []
        for i in range(args.ma_n_conv1d):
            self.ma_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.ma_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.ma_conv1d_layers.append(kL.Dropout(args.dropout))
            self.ma_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.sa_attention_layers = []
        for i in range(args.sa_n_attention):
            self.sa_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.sa_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)

        self.pc_attention_layers = []
        for i in range(args.pc_n_attention):
            self.pc_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.pc_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.mc_attention_layers = []
        for i in range(args.mc_n_attention):
            self.mc_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.mc_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.mg_attention_layers = []
        for i in range(args.mg_n_attention):
            self.mg_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.mg_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.td_attention_layers = []
        for i in range(args.td_n_attention):
            self.td_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.td_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.ma_attention_layers = []
        for i in range(args.ma_n_attention):
            self.ma_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.ma_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.cat_attention_layers = []
        for i in range(args.cat_n_attention):
            self.cat_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                    key_dim=args.mhatt_key_dim, 
                                                                    dropout=args.dropout), 
                                              kL.LayerNormalization()])
        self.cat_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
        self.dense_layers = []
        self.dense_layers.append(kL.Dense(1024, activation=args.activation))
        self.dense_layers.append(kL.Dropout(args.dropout))
        self.dense_layers.append(kL.Dense(512, activation=args.activation))
        self.dense_layers.append(kL.Dense(256, activation=args.activation))
        self.dense_layers.append(kL.Dropout(args.dropout))
        self.dense_layers.append(kL.Dense(32, activation=args.activation))
        self.dense_layers.append(kL.Dense(8, activation=args.activation))
        
        self.dense = kL.Dense(1, activation="linear")
    
    def call(self, xs):
        
        sa, pc, mc, mg, td = xs
        if prints: print('sa, pc, mc, mg, td ')
        if prints: print(sa.shape, pc.shape, mc.shape, mg.shape, td.shape)
        pc = tf.cast(kL.Reshape((-1, 1))(pc), tf.float32)
        mc = tf.cast(kL.Reshape((-1, 1))(mc), tf.float32)
        mg = tf.cast(kL.Reshape((-1, 1))(mg), tf.float32)
        td = tf.cast(kL.Reshape((-1, 1))(td), tf.float32)
        if prints: print('sa, pc, mc, mg, td ')
        if prints: print(sa.shape, pc.shape, mc.shape, mg.shape, td.shape)
            
        for conv1d in self.sa_pre_conv1d_layers:
            sa = conv1d(sa)
            if prints: print('sa_pre_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], sa.shape)
        for conv1d in self.sa_conv1d_layers:
            sa = conv1d(sa)
            if prints: print('sa_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], sa.shape)
                
        sa = kL.Permute((2, 1))(sa)
        if prints: print('permute')
        if prints: print(sa.shape)
        for mhattention, layernorm in self.sa_attention_layers:
            saatt = mhattention(sa, sa)
            sa = layernorm(sa + saatt)
            if prints: print('sa_attention_layers')
            if prints: print(sa.shape)
        sa = kL.Permute((2, 1))(sa)
        if prints: print('permute')
        if prints: print(sa.shape)
        sa = self.sa_attention_dense(sa)
        if prints: print('sa_attention_dense')
        if prints: print(sa.shape)
            
        for conv1d in self.pc_conv1d_layers:
            pc = conv1d(pc)
            if prints: print('pc_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], sa.shape)
            
        pc = kL.Permute((2, 1))(pc)
        if prints: print('permute')
        if prints: print(pc.shape)
        for mhattention, layernorm in self.pc_attention_layers:
            pcatt = mhattention(pc, pc)
            pc = layernorm(pc + pcatt)
            if prints: print('pc_attention_layers')
            if prints: print(pc.shape)
        pc = kL.Permute((2, 1))(pc)
        if prints: print('permute')
        if prints: print(pc.shape)
        pc = self.pc_attention_dense(pc)
        if prints: print('pc_attention_dense')
        if prints: print(pc.shape)
        
        for conv1d in self.mc_conv1d_layers:
            mc = conv1d(mc)
            if prints: print('mc_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], mc.shape)
        
        mc = kL.Permute((2, 1))(mc)
        if prints: print('permute')
        if prints: print(mc.shape)
        for mhattention, layernorm in self.mc_attention_layers:
            mcatt = mhattention(mc, mc)
            mc = layernorm(mc + mcatt)
            if prints: print('mc_attention_layers')
            if prints: print(mc.shape)
        mc = kL.Permute((2, 1))(mc)
        if prints: print('permute')
        if prints: print(mc.shape)
        mc = self.mc_attention_dense(mc)
        if prints: print('mc_attention_dense')
        if prints: print(mc.shape)
        
        for conv1d in self.mg_conv1d_layers:
            mg = conv1d(mg)
            if prints: print('mg_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], mg.shape)
        
        mg = kL.Permute((2, 1))(mg)
        if prints: print('permute')
        if prints: print(mg.shape)
        for mhattention, layernorm in self.mg_attention_layers:
            mgatt = mhattention(mg, mg)
            mg = layernorm(mg + mgatt)
            if prints: print('mg_attention_layers')
            if prints: print(mg.shape)
        mg = kL.Permute((2, 1))(mg)
        if prints: print('permute')
        if prints: print(mg.shape)
        mg = self.mg_attention_dense(mg)
        if prints: print('mg_attention_dense')
        if prints: print(mg.shape)
        
        for conv1d in self.td_conv1d_layers:
            td = conv1d(td)
            if prints: print('td_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], td.shape)
        
        td = kL.Permute((2, 1))(td)
        if prints: print('permute')
        if prints: print(td.shape)
        for mhattention, layernorm in self.td_attention_layers:
            tdatt = mhattention(td, td)
            td = layernorm(td + tdatt)
            if prints: print('td_attention_layers')
            if prints: print(td.shape)
        td = kL.Permute((2, 1))(td)
        if prints: print('permute')
        if prints: print(td.shape)
        td = self.td_attention_dense(td)
        if prints: print('td_attention_dense')
        if prints: print(td.shape)
        
        if prints: print('sa, pc, mc, mg, td')
        if prints: print(sa.shape, pc.shape, mc.shape, mg.shape, td.shape)
        ma = kL.Concatenate(axis=-2)([pc, mc, mg, td])
        if prints: print('ma')
        if prints: print(ma.shape)
        for conv1d in self.ma_conv1d_layers:
            ma = conv1d(ma)
            if prints: print('ma_conv1d_layers')
            if prints: print(str(conv1d).split(' object')[0].split('.')[-1], ma.shape)
            
        ma = kL.Permute((2, 1))(ma)
        if prints: print('permute')
        if prints: print(ma.shape)
        for mhattention, layernorm in self.ma_attention_layers:
            maatt = mhattention(ma, ma)
            ma = layernorm(ma + maatt)
            if prints: print('ma_attention_layers')
            if prints: print(ma.shape)
        ma = kL.Permute((2, 1))(ma)
        if prints: print('permute')
        if prints: print(ma.shape)
        ma = self.ma_attention_dense(ma)
        if prints: print('ma_attention_dense')
        if prints: print(ma.shape)
        x = kL.Concatenate(axis=-2)([sa, ma])
        if prints: print('x')
        if prints: print(x.shape)
        
        x = kL.Permute((2, 1))(x)
        if prints: print('permute')
        if prints: print(x.shape)
        for mhattention, layernorm in self.cat_attention_layers:
            xatt = mhattention(x, x)
            x = layernorm(x + xatt)
            if prints: print('cat_attention_layers')
            if prints: print(x.shape)
        x = kL.Permute((2, 1))(x)
        if prints: print('permute')
        if prints: print(x.shape)
        x = self.cat_attention_dense(x)
        if prints: print('cat_attention_dense')
        if prints: print(x.shape)
        x = kL.Flatten()(x)
        if prints: print('x')
        if prints: print(x.shape)
         
        for dense in self.dense_layers:
            x = dense(x)
            if prints: print('dense_layers')
            if prints: print(x.shape)
        fx = self.dense(x)
        if prints: exit()
        return fx
