import argparse
# from model.tester import model_test
from model.trainer import model_train
from data_loader.data_utils import *
from utils.math_graph import *
import tensorflow as tf
from os.path import join as pjoin
import os

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

if tf.test.is_built_with_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def params():
    num_blocks = Integer(low=2, high=8, name='num_blocks')
    learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
    optim = Categorical(categories=['Adam','RMSprop'], name='optim')
    ks = Categorical(categories=[3,5,7], name='ks')
    kt = Categorical(categories=[3,5,7], name='kt')

    dimensions = [num_blocks,learning_rate,optim,ks,kt]
    default_parameters = [3,1e-3,'RMSprop',3,3]

    return dimensions, default_parameters

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

global least_mae
least_mae = np.inf

# @use_named_args
def fitness(a):
    num_blocks, learning_rate, optim, ks, kt = a
    print('blocks:', num_blocks)
    print('learning rate: ',learning_rate)
    print('optimizer: ', optim)
    print('Ks: ', ks)
    print('Kt: ', kt)
    print()

    n = 22
    n_his = num_blocks*2*(kt-1)+2
    n_pred = 1

    args = Namespace(
        n_route = 22,
        n_his = num_blocks*2*(kt-1)+2,
        n_pred = 1,
        ks = ks,
        kt = kt,
        batch_size = 64,
        epochs = 75,
        datafile = "Delhi_PM2.5_new.csv",
        graph = 'PollutionW_km2_new.csv',
        opt = optim,
        lr = learning_rate,
        inf_mode = ""
    )

    if num_blocks == 2:
        blocks = [[1, 8, 16], [16, 32, 32]]
    elif num_blocks == 3:
        blocks = [[1, 8, 16], [16, 32, 32], [32, 64, 64]]
    elif num_blocks == 4:
        blocks = [[1, 8, 16], [16, 16, 32], [32, 64, 64], [64, 128, 128]]
    elif num_blocks == 5:
        blocks = [[1, 8, 16], [16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128, 256]]
    elif num_blocks == 6:
        blocks = [[1, 8, 16], [16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 256]]
    elif num_blocks == 7:
        blocks = [[1, 8, 16], [16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 512]]
    elif num_blocks == 8:
        blocks = [[1, 8, 16], [16, 16, 32], [32, 32, 64], [64, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512], [512, 1024, 1024]]

    W = weight_matrix(pjoin('./dataset', args.graph), scaling=False)
    # Calculate graph kernel
    L = scaled_laplacian(W)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, ks, n)

    PeMS = data_gen(pjoin('./dataset', args.datafile), n, n_his + n_pred, 0)
    
    try:
        test_loss = model_train(PeMS, Lk, blocks, args)
    except Exception as e:
        print(e)
        test_loss = np.inf

    global least_mae

    if test_loss < least_mae:
        least_mae = test_loss

    tf.keras.backend.clear_session()
    return test_loss


parser = argparse.ArgumentParser()
parser.add_argument('--n_calls', type=int, default=50)
args = parser.parse_args()

dimensions, default_parameters = params()
search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=args.n_calls, x0=default_parameters, verbose=True)
print(search_result.x)