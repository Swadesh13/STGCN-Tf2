import tensorflow as tf
import tensorflow.keras as keras
from .layers import STConvBlock, OutputLayer

class STGCN_Model(keras.Model):
    '''
    Spatio-Temporal Graph Convolutional Neural Model.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param input_shape: list, [time_step, n_route, c_in].
    :param batch_size: int, Batch Size.
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :param dropout: float, dropout ratio.
    :param pad: string, Temporal layer padding - VALID or SAME.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, pad = "VALID", **kwargs):
        super(STGCN_Model, self).__init__(name = "STGCN" ,**kwargs)
        self.n_his = n_his
        self.stconv_blocks = []
        Ko = n_his

        # ST Blocks
        for channels in blocks:
            self.stconv_blocks.append(STConvBlock(graph_kernel, Ks, Kt, channels, act_func, norm, dropout, pad))
            if pad == "VALID":
                Ko -= 2 * (Kt - 1)
        # Output Layer
        if Ko > 1:
            self.output_layer = OutputLayer(Ko, input_shape[1], blocks[-1][-1], blocks[0][0], act_func, norm)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    @tf.function
    def call(self, x:tf.Tensor):
        inputs = x
        x = tf.cast(inputs[:, :self.n_his, :, :], tf.float64)
        for block in self.stconv_blocks:
            x = block(x)
        y = self.output_layer(x)
        return y

    def model(self): # To get brief summary
        x = keras.Input(shape=(21, 22, 1), batch_size=1)
        return keras.Model(inputs=[x], outputs=self.call(x))