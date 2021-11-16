from tensorflow.python.keras.backend import gradients
from data_loader.data_utils import Dataset, gen_batch
from model.model import STGCN_Model
from os.path import join as pjoin

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import tqdm
import math

from utils.math_utils import evaluation


def custom_loss(y_true, y_pred):
    return tf.nn.l2_loss((y_true - y_pred))

def model_train(inputs: Dataset, graph_kernel, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epochs, inf_mode, opt = args.batch_size, args.epochs, args.inf_mode, args.opt

    model = STGCN_Model(inputs.get_data("train").shape[1:], batch_size, graph_kernel, n_his, Ks, Kt, blocks, act_func="GLU", norm="layer", dropout=0.1)
    if opt == "RMSprop":
        optimizer = keras.optimizers.RMSprop(args.lr)
    elif opt == "Adam":
        optimizer = keras.optimizers.Adam(args.lr)
    elif opt == "SGD":
        optimizer = keras.optimizers.SGD(args.lr)
    else:
        raise NotImplementedError(f'ERROR: optimizer "{opt}" is not implemented')
    
    # print("Training Model on Data")
    # # for epoch in range(epochs):
    # for epoch in range(4):
    #     print(f"\nEpoch {epoch}")
    #     mae_loss, mape_loss, rmse_loss = 0, 0, 0
    #     for batch in tqdm.tqdm(gen_batch(inputs.get_data("train")[:, :, :, :], batch_size, dynamic_batch=True, shuffle=True), total=math.ceil(inputs.get_data("train").shape[0]/batch_size)): #math.ceil(inputs.get_data("train").shape[0]/batch_size)
    #         with tf.GradientTape() as tape:
    #             y_pred = model(batch[:, :n_his, :, :], training=True)
    #             loss = mae(batch[:, n_his:n_his+1, :, :], y_pred)
    #             mae_loss += mae(batch[:, n_his:n_his+1, :, :], y_pred)
    #             mape_loss += mape(batch[:, n_his:n_his+1, :, :], y_pred)
    #             rmse_loss += rmse(batch[:, n_his:n_his+1, :, :], y_pred)
    #             # print(y_pred[0, :, :]*inputs.std + inputs.mean)
    #             # print(batch[0, n_his:n_his+1, :, :]*inputs.std + inputs.mean)

    #         gradients = tape.gradient(loss, model.trainable_weights)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    #     print("MAE Loss: ", mae_loss)
    #     print(f"Epoch {epoch} finished!")

    model.compile(optimizer=optimizer, loss=custom_loss)
    model.fit(inputs.get_data("train")[:, :n_his, :, :], inputs.get_data("train")[:, n_his:n_his+1, :, :], 
            validation_data=(inputs.get_data("val")[:, :n_his, :, :], inputs.get_data("val")[:, n_his:n_his+1, :, :]),
            epochs=epochs, batch_size=batch_size)

    print(np.array(inputs.get_data("test")[:1, n_his:n_his+1, :, :], dtype=np.float)*inputs.std+inputs.std)
    print(model(inputs.get_data("test")[:1, :n_his, :, :])*inputs.std+inputs.mean)

    x_test = inputs.get_data("test")[:, :n_his, :, :]
    y_test = inputs.get_data("test")[:, n_his:n_his+1, :, :]
    preds = model(x_test)

    print(evaluation(y_test[0], preds[0], inputs.get_stats()))