import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dtype


class TemporalConvLayer(keras.layers.Layer):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :param pad: string, Temporal layer padding - VALID or SAME.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    def __init__(self, Kt, c_in, c_out, act_func='relu', pad='VALID'):
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        self.pad = pad
    
    def build(self, input_shape):
        if self.c_in > self.c_out:
            self.down_sample_conv_weights = self.add_weight(name="down_sample_conv_weights", shape=[1,1,self.c_in,self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        if self.act_func == "GLU":
            c_out = 2*self.c_out
        else:
            c_out = self.c_out
        self.dense_weights = self.add_weight(name="dense_weights", shape=[self.Kt, 1, self.c_in, c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_bias =  self.add_weight(name="dense_bias", shape=[c_out], dtype=tf.float64, trainable=True)

    @tf.function
    def call(self, x: tf.Tensor):
        _, T, n, _ = x.shape
        x = tf.cast(x, tf.float64)

        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = tf.nn.conv2d(x, self.down_sample_conv_weights, strides=[1]*4, padding="SAME")
        elif self.c_in < self.c_out:
            x_input = tf.concat([x, tf.zeros(shape=[tf.shape(x)[0], T, n, self.c_out - self.c_in], dtype=tf.float64)], axis=3)
        else:
            x_input = x

        # keep the original input for residual connection.
        if self.pad == 'VALID':
            x_input = x_input[:, self.Kt - 1:T, :, :]

        x_conv = tf.nn.conv2d(x, self.dense_weights, strides=[1]*4, padding=self.pad) + self.dense_bias

        if self.act_func == "GLU":
            return (x_conv[:,:,:,:self.c_out] + x_input) * tf.nn.sigmoid(x_conv[:,:,:,self.c_out:])
        elif self.act_func == "linear":
            return x_conv
        elif self.act_func == "sigmoid":
            return tf.nn.sigmoid(x_conv)
        elif self.act_func == "relu":
            return tf.nn.relu(x_conv + x_input)
        else:
            raise NotImplementedError(f'ERROR: activation function "{self.act_func}" is not implemented.')


class SpatioConvLayer(keras.layers.Layer):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, graph_kernel, Ks, c_in, c_out):
        super().__init__()
        self.graph_kernel = tf.Variable(initial_value = graph_kernel, trainable=False)
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out

    def build(self, input_shape):
        if self.c_in > self.c_out:
            self.down_sample_conv_weights = self.add_weight(name="down_sample_conv_weights", shape=[1,1,self.c_in,self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_weights = self.add_weight(name="dense_weights", shape=[self.Ks*self.c_in, self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_bias =  self.add_weight(name="dense_bias", shape=[self.c_out], dtype=tf.float64, trainable=True)

    @tf.function
    def call(self, x: tf.Tensor):
        _, T, n, _ = x.shape
        x = tf.cast(x, tf.float64)

        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = tf.nn.conv2d(x, self.down_sample_conv_weights, strides=[1]*4, padding="SAME")
        elif self.c_in < self.c_out:
            x_input = tf.concat([x, tf.zeros(shape=[tf.shape(x)[0], T, n, self.c_out - self.c_in], dtype=tf.float64)], axis=3)
        else:
            x_input = x

        # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
        x = tf.reshape(x, [-1, n, self.c_in])
        # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
        x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
        # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
        x_mul = tf.reshape(tf.matmul(x_tmp , self.graph_kernel), [-1, self.c_in, self.Ks, n])
        # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
        x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, self.c_in * self.Ks])
        # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
        x_gconv = tf.reshape(tf.matmul(x_ker, self.dense_weights), [-1, n, self.c_out]) + self.dense_bias
        # x_gconv -> [batch_size, time_step, n_route, c_out]
        x_gconv = tf.reshape(x_gconv, [-1, T, n, self.c_out])
        out = x_gconv[:,:,:,:self.c_out] + x_input
        return tf.nn.relu(out)


class FullyConLayer(layers.Layer):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param outc: int, output channel size.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    def __init__(self, n, channel, outc):
        super().__init__()
        self.n = n
        self.channel = channel
        self.outc = outc

    def build(self, input_shape):
        self.dense_weights = self.add_weight(name="dense_weights", shape=[1, 1, self.channel, self.outc], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_bias =  self.add_weight(name="dense_bias", shape=[self.n, self.outc], dtype=tf.float64, trainable=True)
    
    @tf.function
    def call(self, x: tf.Tensor):
        x = tf.cast(x, tf.float64)
        return tf.nn.conv2d(x, self.dense_weights, strides=[1, 1, 1, 1], padding='SAME') + self.dense_bias


class OutputLayer(keras.layers.Layer):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param Kt: int, kernel size of temporal convolution.
    :param channel: int, input channel size.
    :param outc: int, output channel size.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    def __init__(self, Kt, n, channel, outc=1, act_func="GLU", norm="layer"):
        super().__init__()
        self.Kt = Kt
        self.n = n
        self.outc = outc
        self.act_func = act_func
        self.norm = norm
        self.layer1 = TemporalConvLayer(self.Kt, channel, channel, self.act_func)
        self.layer2 = TemporalConvLayer(1, channel, channel, self.act_func)
        self.layer3 = FullyConLayer(n, channel, outc)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization(axis=[2,3])
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')

    @tf.function
    def call(self, x:tf.Tensor):
        x_i = self.layer1(x)
        if self.norm == "L2":
            x_ln = tf.nn.l2_normalize(x_i, axis=[2,3])
        else:
            x_ln = self.normalization(x_i)
        x_o = self.layer2(x_ln)
        fc = self.layer3(x_o)
        return tf.reshape(fc, shape=[-1, 1, self.n, self.outc])


class STConvBlock(keras.layers.Layer):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :param dropout: float, dropout ratio.
    :param pad: string, Temporal layer padding - VALID or SAME.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, graph_kernel, Ks, Kt, channels, act_func='GLU', norm='layer', dropout=0.2, pad='VALID'):
        super().__init__()
        self.norm = norm
        c_si, c_t, c_oo = channels
        n = graph_kernel.shape[0]
        self.layer1 = TemporalConvLayer(Kt, c_si, c_t, act_func, pad)
        self.layer2 = SpatioConvLayer(graph_kernel, Ks, c_t, c_t)
        self.layer3 = TemporalConvLayer(Kt, c_t, c_oo, act_func, pad)
        self.dropout_layer = keras.layers.Dropout(rate = dropout)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization(axis=[2,3])
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')

    @tf.function
    def call(self, x:tf.Tensor):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        if self.norm == "L2":
            out = tf.nn.l2_normalize(x3, axis=[2,3])
        else:
            out = self.normalization(x3)
        return self.dropout_layer(out)