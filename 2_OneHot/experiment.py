import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import callbacks
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import datetime
import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import dataset
import model


def results(true_list, pred_list):
    r2 = r2_score(true_list, pred_list)
    rmse = math.sqrt(mean_squared_error(true_list, pred_list))
    mae = mean_absolute_error(true_list, pred_list)
    print("- R2 : {:.4f}".format(r2))
    print("- RMSE : {:.4f}".format(rmse))
    print("- MAE : {:.4f}".format(mae))
    return r2, rmse, mae

def experiment(args, train_data, test_data):
    
    train_xki, train_xpc, train_xmc, train_xmg, train_ytg = train_data
    test_xki, test_xpc, test_xmc, test_xmg, test_ytg = test_data
    
    my_strategy = tf.distribute.MirroredStrategy()
    with my_strategy.scope():
        kpro = model.KPro(args)
        lr_schedule = ExponentialDecay(initial_learning_rate=args.lr, 
                                       decay_steps=args.decay_steps, 
                                       decay_rate=args.decay_rate)
        kpro.compile(loss=[MeanSquaredError()], 
                     optimizer=Adam(learning_rate=lr_schedule), 
                     metrics=[RootMeanSquaredError()])

    print('[ Train & Valid ]')
    log_dir = str(args.log) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    e_stop = callbacks.EarlyStopping(monitor='val_loss', 
                                     min_delta=0.0, 
                                     patience=args.patience, 
                                     verbose=0, 
                                     mode='auto')
    tensorboard = callbacks.TensorBoard(log_dir=log_dir)
    history = kpro.fit([train_xki, train_xpc, train_xmc, train_xmg], train_ytg, 
                       epochs=args.epochs,
                       batch_size=args.batch,
                       validation_data=([test_xki, test_xpc, test_xmc, test_xmg], test_ytg), 
                       callbacks=[e_stop, tensorboard])
    print()
    train_loss_list = history.history['loss']
    valid_loss_list = history.history['val_loss']
    train_rmse_list = history.history['root_mean_squared_error']
    valid_rmse_list = history.history['val_root_mean_squared_error']
    
    print('[ Test ]')
    pred_list = kpro.predict([test_xki, test_xpc, test_xmc, test_xmg], batch_size=args.batch)
    true_list = test_ytg
    pred_list = pred_list.reshape(-1)
    
    r2, rmse, mae = results(true_list, pred_list)
    
    args.train_loss_list = train_loss_list
    args.valid_loss_list = valid_loss_list
    args.train_rmse_list = train_rmse_list
    args.valid_rmse_list = valid_rmse_list
    args.true_list = true_list
    args.pred_list = pred_list
    args.r2 = r2
    args.rmse = rmse
    args.mae = mae
    
    stringlist = []
    kpro.summary(print_fn=lambda x: stringlist.append(x))
    args.model_summary = "\n".join(stringlist)
    return args, kpro