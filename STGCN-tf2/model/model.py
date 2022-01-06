import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from .layers import STConvBlock, OutputLayer, SConvBlock, TConvBlock

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
    def __init__(self, input_shape, batch_size, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, pad = "VALID", **kwargs):
        super(STGCN_Model, self).__init__(name = "STGCN" ,**kwargs)
        self.n_his = n_his
        self.stconv_blocks = []
        Ko = n_his

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float64)
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

    def call(self, x:tf.Tensor):
        inputs = x
        x = tf.cast(inputs[:, :self.n_his, :, :], tf.float64)
        x = self.input_layer(x)
        for block in self.stconv_blocks:
            x = block(x)
        y = self.output_layer(x)
        return y

    def model(self): # To get brief summary
        x = keras.Input(shape=(21, 22, 1), batch_size=1)
        return keras.Model(inputs=[x], outputs=self.call(x))


class STGCNB_Model(keras.Model):
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
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, input_shape, batch_size, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(STGCNB_Model, self).__init__(name = "STGCNB" ,**kwargs)
        self.n_his = n_his
        self.stconv_blocks = []

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float64)
        # ST Blocks
        for channels in blocks:
            self.stconv_blocks.append([TConvBlock(Kt, channels, act_func, norm, dropout), SConvBlock(graph_kernel, Ks, channels, norm, dropout)])
        # Output Layer
        self.output_layer = OutputLayer(n_his, input_shape[1], blocks[-1][-1]*2, blocks[0][0], act_func, norm)

    def call(self, x:tf.Tensor):
        inputs = x
        x = tf.cast(inputs[:, :self.n_his, :, :], tf.float64)
        x = self.input_layer(x)
        for block in self.stconv_blocks:
            x1 = block[0](x)
            x2 = block[1](x)
            x = tf.concat([x1, x2], axis=-1)

        y = self.output_layer(x)
        return y

    def model(self): # To get brief summary
        x = keras.Input(shape=(20, 20, 1), batch_size=1)
        return keras.Model(inputs=[x], outputs=self.call(x))