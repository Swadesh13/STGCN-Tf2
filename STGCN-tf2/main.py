import argparse
# from model.tester import model_test
from model.trainer import model_train
from data_loader.data_utils import *
from utils.math_graph import *
import tensorflow as tf
from os.path import join as pjoin
import os

if tf.test.is_built_with_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=22)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSprop')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--datafile', type=str, default=f'Delhi_PM2.5_new.csv')
parser.add_argument('--graph', type=str, default='PollutionW_km2_new.csv')
parser.add_argument('--channels', type=int, default=1)

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[args.channels, 32, 64], [64, 64, 64], [64, 128, 128]]
# blocks = [[args.channels, 32, 32], [64, 64, 64], [128, 128, 128]] # for STGCN-B

W = weight_matrix(pjoin('./dataset', args.graph), scaling=False)
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
# tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float64))

PeMS = data_gen(pjoin('./dataset', args.datafile), n, n_his + n_pred, args.channels)
# print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

# print(PeMS.get_data("train").shape)

if __name__ == '__main__':
    model_train(PeMS, Lk, blocks, args)
    # model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
