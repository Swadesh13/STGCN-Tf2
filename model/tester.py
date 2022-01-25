from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, custom_loss
from os.path import join as pjoin

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time


def multi_pred(model, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param model: tf.keras Model.
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test = np.copy(i[:, 0:n_his, :, :])
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for _ in range(n_pred):
            pred = model(test)
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = np.reshape(pred, (test_seq.shape[0], test_seq.shape[2], test_seq.shape[3]))
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    shape = pred_array.shape
    pred_array = np.reshape(pred_array, (shape[0], shape[1], *shape[3:]))
    return pred_array[step_idx], pred_array.shape[1]


def model_inference(model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param model: tf.keras Model.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val = multi_pred(model, x_val, batch_size, n_his, n_pred, step_idx)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(model, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred
    return min_va_val, min_val


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model = keras.models.load_model(load_path, {"custom_loss": custom_loss})
    print(f'>> Loading saved model from {load_path} ...')

    if inf_mode == 'sep':
        # for inference mode 'sep', the type of step index is int.
        step_idx = n_pred - 1
        tmp_idx = [step_idx]
    elif inf_mode == 'merge':
        # for inference mode 'merge', the type of step index is np.ndarray.
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
    else:
        raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

    x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

    y_test, len_test = multi_pred(model, x_test, batch_size, n_his, n_pred, step_idx)
    evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)

    for ix in tmp_idx:
        te = evl[ix - 2:ix + 1]
        print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
    print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')