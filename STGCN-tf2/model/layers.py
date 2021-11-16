import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dtype


class TemporalConvLayer(keras.layers.Layer):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
    
    def build(self, input_shape):
        if self.c_in>self.c_out:
            self.down_sample_conv_weights = self.add_weight(name="down_sample_conv_weights", shape=[1,1,self.c_in,self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
            #self.add_loss(tf.nn.l2_loss(self.down_sample_conv_weights))
        if self.act_func == "GLU":
            self.dense_weights = self.add_weight(name="dense_weights", shape=[self.Kt, 1, self.c_in, 2*self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
            self.dense_bias =  self.add_weight(name="dense_bias", shape=[2*self.c_out], dtype=tf.float64, trainable=True)
        else:
            self.dense_weights = self.add_weight(name="dense_weights", shape=[self.Kt, 1, self.c_in, self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
            self.dense_bias =  self.add_weight(name="dense_bias", shape=[self.c_out], dtype=tf.float64, trainable=True)
        #self.add_loss(tf.nn.l2_loss(self.dense_weights))

    def call(self, x: tf.Tensor):
        _, T, n, _ = x.shape
        x = tf.cast(x, tf.float64)

        if self.c_in>self.c_out:
            x_input=tf.nn.conv2d(x, self.down_sample_conv_weights, strides=[1]*4, padding="SAME")
        elif self.c_in<self.c_out:
            x_input=tf.concat([x, tf.zeros(shape=[tf.shape(x)[0], T, n, self.c_out - self.c_in], dtype=tf.float64)], axis=3)
        else:
            x_input=x

        x_input = x_input[:, self.Kt - 1:T, :, :]
        x_conv = tf.nn.conv2d(x, self.dense_weights, strides=[1, 1, 1, 1], padding='VALID') + self.dense_bias
        
        if self.act_func == "GLU":
            return (x_conv[:,:,:,:self.c_out] + x_input) * tf.nn.sigmoid(x_conv[:,:,:,-self.c_out:])
        elif self.act_func == "linear":
            return x_conv
        elif self.act_func == "sigmoid":
            return tf.nn.sigmoid(x_conv)
        elif self.act_func == "relu":
            return tf.nn.relu(x_conv + x_input)
        else:
            raise NotImplementedError(f'ERROR: activation function "{self.act_func}" is not implemented.')


class SpatioConvLayer(keras.layers.Layer):
    def __init__(self, graph_kernel, Ks, c_in, c_out):
        super().__init__()
        self.graph_kernel = tf.Variable(initial_value = graph_kernel, trainable=False)
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out

    def build(self, input_shape):
        if self.c_in>self.c_out:
            self.down_sample_conv_weights = self.add_weight(name="down_sample_conv_weights", shape=[1,1,self.c_in,self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
            #self.add_loss(tf.nn.l2_loss(self.down_sample_conv_weights))
        self.dense_weights = self.add_weight(name="dense_weights", shape=[self.Ks*self.c_in, self.c_out], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_bias =  self.add_weight(name="dense_bias", shape=[self.c_out], dtype=tf.float64, trainable=True)
        #self.add_loss(tf.nn.l2_loss(self.dense_weights))

    def call(self, x: tf.Tensor):
        _, T, n, _ = x.shape
        x = tf.cast(x, tf.float64)

        if self.c_in>self.c_out:
            x_input=tf.nn.conv2d(x, self.down_sample_conv_weights, strides=[1]*4, padding="SAME")
        elif self.c_in<self.c_out:
            x_input=tf.concat([x, tf.zeros(shape=[tf.shape(x)[0], T, n, self.c_out - self.c_in])], axis=3)
        else:
            x_input=x

        x = tf.reshape(x, [-1, n, self.c_in])
        x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
        x_mul = tf.reshape(tf.matmul(x_tmp , self.graph_kernel), [-1, self.c_in, self.Ks, n])
        x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, self.c_in * self.Ks])
        x_gconv = tf.reshape(tf.matmul(x_ker, self.dense_weights), [-1, n, self.c_out]) + self.dense_bias
        x_gconv = tf.reshape(x_gconv, [-1, T, n, self.c_out])
        out = x_gconv[:,:,:,:self.c_out] + x_input
        return tf.nn.relu(out)


class FullyConLayer(layers.Layer):
    def __init__(self, n, channel):
        super().__init__()
        self.n = n
        self.channel = channel

    def build(self, input_shape):
        self.dense_weights = self.add_weight(name="dense_weights", shape=[1, 1, self.channel, 1], dtype=tf.float64, initializer='glorot_uniform', trainable=True)
        self.dense_bias =  self.add_weight(name="dense_bias", shape=[self.n, 1], dtype=tf.float64, trainable=True)
        #self.add_loss(tf.nn.l2_loss(self.dense_weights))

    def call(self, x: tf.Tensor):
        x = tf.cast(x, tf.float64)
        return tf.nn.conv2d(x, self.dense_weights, strides=[1, 1, 1, 1], padding='SAME') + self.dense_bias


class STConvBlock(keras.layers.Layer):
    def __init__(self, graph_kernel, Ks, Kt, channels, act_func='GLU', norm="L2", dropout=0.2):
        super().__init__()
        self.dropout_layer = keras.layers.Dropout(rate = dropout)
        self.norm = norm
        c_si, c_t, c_oo = channels
        n = graph_kernel.shape[0]
        self.layer1 = TemporalConvLayer(Kt, c_si, c_t, act_func)
        self.layer2 = SpatioConvLayer(graph_kernel, Ks, c_t, c_t)
        self.layer3 = TemporalConvLayer(Kt, c_t, c_oo, act_func)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization()
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        else:
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')

    def call(self, x:tf.Tensor):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        if self.norm == "L2":
            out = tf.nn.l2_normalize(x3, axis=[2,3])
        else:
            out = self.normalization(x3)
        return self.dropout_layer(out)


class OutputLayer(keras.layers.Layer):
    def __init__(self, Kt, n, channel, act_func="GLU", norm="L2"):
        super().__init__()
        self.Kt = Kt
        self.act_func = act_func
        self.norm = norm
        self.layer1 = TemporalConvLayer(self.Kt, channel, channel, self.act_func)
        self.layer2 = TemporalConvLayer(1, channel, channel, self.act_func)
        self.layer3 = FullyConLayer(n, channel)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization()
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        else:
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')
    
    def call(self, x:tf.Tensor):
        n = x.shape[-2]
        x_i = self.layer1(x)
        if self.norm == "L2":
            x_ln = tf.nn.l2_normalize(x_i, axis=[2,3])
        else:
            x_ln = self.normalization(x_i)
        x_o = self.layer2(x_ln)
        fc = self.layer3(x_o)
        return tf.reshape(fc, shape=[-1, 1, n, 1])