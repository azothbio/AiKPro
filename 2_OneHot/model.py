import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as kL
from tensorflow.keras import models as kM
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

prints = False

class KPro(Model):
    
    def __init__(self, args):
        super().__init__()
        
        self.ki_conv1d_layers = []
        for i in range(args.ki_n_conv1d):
            self.ki_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.ki_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.ki_conv1d_layers.append(kL.Dropout(args.dropout))
            self.ki_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
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
                
        self.ma_conv1d_layers = []
        for i in range(args.ma_n_conv1d):
            self.ma_conv1d_layers.append(kL.Conv1D(filters=args.conv_1dfilters, 
                                                   kernel_size=args.conv_1dkernel, 
                                                   strides=args.conv_1dstrides, 
                                                   activation=args.activation))
            self.ma_conv1d_layers.append(kL.MaxPool1D(pool_size=args.pool1d))
            self.ma_conv1d_layers.append(kL.Dropout(args.dropout))
            self.ma_conv1d_layers.append(kL.Dense(16, activation=args.activation))
        
        self.ki_attention_layers = []
        for i in range(args.ki_n_attention):
            self.ki_attention_layers.append([kL.MultiHeadAttention(num_heads=args.mhatt_num_heads, 
                                                                   key_dim=args.mhatt_key_dim, 
                                                                   dropout=args.dropout), 
                                             kL.LayerNormalization()])
        self.ki_attention_dense = kL.Dense(args.conv_1dfilters, activation=args.activation)
        
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
        
        ki, pc, mc, mg = xs
        if prints: print('ki, pc, mc, mg')
        if prints: print(ki.shape, pc.shape, mc.shape, mg.shape)
        
        ki = tf.cast(kL.Reshape((-1, 1))(ki), tf.float32)
        pc = tf.cast(kL.Reshape((-1, 1))(pc), tf.float32)
        mc = tf.cast(kL.Reshape((-1, 1))(mc), tf.float32)
        mg = tf.cast(kL.Reshape((-1, 1))(mg), tf.float32)
        if prints: print('ki, pc, mc, mg')
        if prints: print(ki.shape, pc.shape, mc.shape, mg.shape)
        for conv1d in self.ki_conv1d_layers:
            ki = conv1d(ki)
        
        ki = kL.Permute((2, 1))(ki)
        for mhattention, layernorm in self.ki_attention_layers:
            kiatt = mhattention(ki, ki)
            ki = layernorm(ki + kiatt)
        ki = kL.Permute((2, 1))(ki)
        ki = self.ki_attention_dense(ki)
        
        for conv1d in self.pc_conv1d_layers:
            pc = conv1d(pc)
        
        pc = kL.Permute((2, 1))(pc)
        for mhattention, layernorm in self.pc_attention_layers:
            pcatt = mhattention(pc, pc)
            pc = layernorm(pc + pcatt)
        pc = kL.Permute((2, 1))(pc)
        pc = self.pc_attention_dense(pc)
        
        for conv1d in self.mc_conv1d_layers:
            mc = conv1d(mc)
        
        mc = kL.Permute((2, 1))(mc)
        for mhattention, layernorm in self.mc_attention_layers:
            mcatt = mhattention(mc, mc)
            mc = layernorm(mc + mcatt)
        mc = kL.Permute((2, 1))(mc)
        mc = self.mc_attention_dense(mc)
        
        for conv1d in self.mg_conv1d_layers:
            mg = conv1d(mg)
        
        mg = kL.Permute((2, 1))(mg)
        for mhattention, layernorm in self.mg_attention_layers:
            mgatt = mhattention(mg, mg)
            mg = layernorm(mg + mgatt)
        mg = kL.Permute((2, 1))(mg)
        mg = self.mg_attention_dense(mg)
        
        if prints: print('ki, pc, mc, mg')
        if prints: print(ki.shape, pc.shape, mc.shape, mg.shape)
        ma = kL.Concatenate(axis=-2)([pc, mc, mg])
        if prints: print('ma')
        if prints: print(ma.shape)
        for conv1d in self.ma_conv1d_layers:
            ma = conv1d(ma)
        
        ma = kL.Permute((2, 1))(ma)
        for mhattention, layernorm in self.ma_attention_layers:
            maatt = mhattention(ma, ma)
            ma = layernorm(ma + maatt)
        ma = kL.Permute((2, 1))(ma)
        ma = self.ma_attention_dense(ma)
        
        if prints: print('ma')
        if prints: print(ma.shape)
        if prints: print('ki')
        if prints: print(ki.shape)
        x = kL.Concatenate(axis=-2)([ki, ma])
        if prints: print('x')
        if prints: print(x.shape)            
        x = kL.Permute((2,1))(x)
        for mhattention, layernorm in self.cat_attention_layers:
            xatt = mhattention(x, x)
            x = layernorm(x + xatt)
        x = kL.Permute((2, 1))(x)
        x = self.cat_attention_dense(x)
        
        if prints: print('x')
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
