import tensorflow as tf
import tensorflow.keras as keras
from .layers import STConvBlock, OutputLayer

class STGCN_Model(keras.Model):
    def __init__(self, input_shape, batch_size, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(STGCN_Model, self).__init__(name = "STGCN" ,**kwargs)
        self.n_his = n_his
        Ko = n_his
        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float64)
        self.stconv_blocks = []

        for channels in blocks:
            self.stconv_blocks.append(STConvBlock(graph_kernel, Ks, Kt, channels, act_func, norm, dropout))
            Ko -= 2 * (Kt - 1)

        if Ko > 1:
            self.output_layer = OutputLayer(Ko, input_shape[1], blocks[-1][-1], act_func, norm)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def call(self, x:tf.Tensor):
        n_his = self.n_his
        inputs = x
        x = tf.cast(inputs[:, 0:n_his, :, :], tf.float64)
        x = self.input_layer(x)
        for block in self.stconv_blocks:
            x = block(x)
        y = self.output_layer(x)
        return y

    def model(self): # To get summary
        x = keras.Input(shape=(20, 20, 1), batch_size=1)
        return keras.Model(inputs=[x], outputs=self.call(x))