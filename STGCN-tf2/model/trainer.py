from data_loader.data_utils import Dataset, gen_batch
from model.model import STGCN_Model
from os.path import join as pjoin
from utils.math_utils import evaluation, MAPE, MAE, RMSE

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import tqdm
import math


def custom_loss(y_true, y_pred) -> tf.Tensor:
    # return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return tf.nn.l2_loss(y_true - y_pred)

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
    train_data = inputs.get_data("train")
    val_data = inputs.get_data("val")
    steps_per_epoch = math.ceil(train_data.shape[0]/batch_size)

    model = STGCN_Model(inputs.get_data("train").shape[1:], batch_size, graph_kernel, n_his, Ks, Kt, blocks, act_func="GLU", norm="layer", dropout=0.1)
    lr_func = keras.optimizers.schedules.PiecewiseConstantDecay(
        [50*steps_per_epoch, 60*steps_per_epoch, 70*steps_per_epoch],
        [args.lr, 0.75*args.lr, 0.5*args.lr, 0.25*args.lr]
    )
    if opt == "RMSprop":
        optimizer = keras.optimizers.RMSprop(lr_func)
    elif opt == "Adam":
        optimizer = keras.optimizers.Adam(lr_func)
    else:
        raise NotImplementedError(f'ERROR: optimizer "{opt}" is not implemented')

    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[keras.metrics.MeanAbsoluteError(name="mae"), keras.metrics.RootMeanSquaredError(name="rmse"), keras.metrics.MeanAbsolutePercentageError(name="mape")])

    print("Training Model on Data")
    best_val_mae = np.inf
    weights_file = f"best_nblocks_{len(blocks)}_nhis_{args.n_his}_opt_{opt}_ks_{Ks}_kt_{Kt}_lr_{args.lr}.h5"
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1} / {epochs}")
        train_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(gen_batch(train_data, batch_size, dynamic_batch=True, shuffle=False), total=steps_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(batch[:, :n_his, :, :], training=True)
                loss = custom_loss(batch[:, n_his:n_his+1, :, :], y_pred)
                gradients = tape.gradient(loss, model.trainable_weights)
            train_loss += loss
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            model.compiled_metrics.update_state(tf.Variable(batch[:, n_his:n_his+1, :, :]*inputs.std+inputs.mean), y_pred*inputs.std+inputs.mean)

        print(f"Epoch {epoch} finished!", "Training Time:", f"{time.time()-start_time}s")
        print("Train L2 Loss: ", train_loss.numpy(), end="\t")
        for m in model.metrics: print(f"train_{m.name}: {m.result()}", end="\t")
        print()
        model.reset_metrics()

        val_train = val_data[:, :n_his, :, :]
        val_preds = model(val_train, training=False)
        model.compiled_metrics.update_state(tf.Variable(val_data[:, n_his:n_his+1, :, :]*inputs.std+inputs.mean), val_preds*inputs.std+inputs.mean)
        val_loss = custom_loss(val_data[:, n_his:n_his+1, :, :], val_preds)
        print("Val L2 Loss: ", val_loss.numpy(), end="\t")
        for m in model.metrics: 
            print(f"val_{m.name}: {m.result()}", end="\t")
            if m.name == "mae" and best_val_mae > m.result():
                best_val_mae = m.result()
                model.save_weights(weights_file)
        print()
        model.reset_metrics()
        
    model.load_weights(weights_file)
    x_test = inputs.get_data("test")[:, :n_his, :, :]
    y_test = inputs.get_data("test")[:, n_his:n_his+1, :, :]
    preds = model(x_test)

    # print(np.array(inputs.get_data("test")[:1, n_his:n_his+1, :, :], dtype=np.float))
    # print(model(inputs.get_data("test")[:1, :n_his, :, :]))

    test_m = evaluation(y_test, preds, inputs.get_stats())
    print("TEST (MAPE, MAE, RMSE):", test_m)
    return float(test_m[1])