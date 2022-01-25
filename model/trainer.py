from data_loader.data_utils import Dataset, gen_batch
from model.model import STGCN_Model
from model.tester import model_inference
from utils.math_utils import custom_loss

import tensorflow as tf
import numpy as np
import time
import tqdm
import math
import os


def model_train(inputs: Dataset, graph_kernel, blocks, args):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param graph_kernel: np.array, [n_route, Ks*n_route].
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epochs, inf_mode, opt = args.batch_size, args.epochs, args.inf_mode, args.opt
    train_data = inputs.get_data("train")[:32*2]
    val_data = inputs.get_data("val")
    steps_per_epoch = math.ceil(train_data.shape[0]/batch_size)

    os.makedirs(args.model_path, exist_ok=True)
    train_log_dir = os.path.join(args.logs, 'train')
    test_log_dir = os.path.join(args.logs, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model = STGCN_Model(train_data.shape[1:], graph_kernel, n_his, Ks, Kt, blocks, "GLU", "layer", 0.1, "VALID")
    lr_func = tf.optimizers.schedules.PiecewiseConstantDecay(
        [50*steps_per_epoch, 60*steps_per_epoch, 70*steps_per_epoch],
        [args.lr, 0.75*args.lr, 0.5*args.lr, 0.25*args.lr]
    )
    if opt == "RMSprop":
        optimizer = tf.optimizers.RMSprop(lr_func)
    elif opt == "Adam":
        optimizer = tf.optimizers.Adam(lr_func)
    else:
        raise NotImplementedError(f'ERROR: optimizer "{opt}" is not implemented')

    print("Training Model on Data")
    best_val_mae = np.inf
    
    if inf_mode == 'sep':
        # for inference mode 'sep', the type of step index is int.
        step_idx = n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5])
    elif inf_mode == 'merge':
        # for inference mode 'merge', the type of step index is np.ndarray.
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
    else:
        raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch} / {epochs}")
        train_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(gen_batch(train_data, batch_size, dynamic_batch=True, shuffle=True), total=steps_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(batch[:, :n_his, :, :])
                loss = custom_loss(batch[:, n_his:n_his+1, :, :], y_pred)
                gradients = tape.gradient(loss, model.trainable_variables())
            train_loss += loss
            optimizer.apply_gradients(zip(gradients, model.trainable_variables()))

        print(f"Epoch {epoch} finished!", "Training Time:", f"{time.time()-start_time}s")
        print("Train L2 Loss: ", f"{train_loss.numpy():.4f}", end="\t")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)

        val_train = val_data[:, :n_his, :, :]
        val_preds = model(val_train)
        val_loss = custom_loss(val_data[:, n_his:n_his+1, :, :], val_preds)
        print("Val L2 Loss: ", f"{val_loss.numpy():.4f}", end="\t")
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
        print()

        tf.saved_model.save(model, os.path.join(args.model_path, str(epoch)))

        # checkpoint = tf.train.Checkpoint(model)
        # checkpoint.save(os.path.join(args.model_path, str(epoch)))
        # print(f"Saving best model at {args.model_path} (based on MAE)")

        start_time = time.time()
        min_va_val, min_val = model_inference(model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

        for ix in tmp_idx:
            va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: '
                    f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                    f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                    f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
        print(f'Epoch {epoch:2d} Inference Time {time.time() - start_time:.3f}s')

    print('Training model finished!')