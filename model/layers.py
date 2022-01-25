import tensorflow as tf

class WB:
    '''
    Keeps track of Weights and Biases of the model as tf.Variable.
    :param name: str, Name of the Variable
    :param shape: tuple, shape of the Variable
    :param dtype: tf.dtype, data type of the Variable
    :param trainable: bool, whether the Variable is trainable or not
    :param initialize_func: function, how to initialize the weight/bias. Must take shape and dtype as argument and return a tf.Tensor.
    '''
    def __init__(self):
        self.weights = []
        self.biases = []

    def get_w(self, name, shape, dtype, trainable, initialize_func = tf.random.truncated_normal):
        w = tf.Variable(initial_value=initialize_func(shape, dtype=dtype), name=name, trainable=trainable)
        self.weights.append(w)
        return w

    def get_b(self, name, shape, dtype, trainable, initialize_func = tf.random.truncated_normal):
        b = tf.Variable(initial_value=initialize_func(shape, dtype=dtype), name=name, trainable=trainable)
        self.biases.append(b)
        return b

    def get_wb(self):
        return self.weights + self.biases


class LayerNorm:
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    def __init__(self, wb: WB, n_route, channel) -> None:
        self.gamma = wb.get_w('gamma', [1, 1, n_route, channel], tf.float64, True, tf.ones)
        self.beta = wb.get_b('beta', [1, 1, n_route, channel], tf.float64, True, tf.zeros)

    @tf.function
    def __call__(self, x):
        mu, sigma = tf.nn.moments(x, axes=[2, 3], keepdims=True)
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * self.gamma + self.beta
        return _x


class TemporalConvLayer:
    def __init__(self, wb: WB, Kt, c_in, c_out, act_func='GLU', pad='VALID'):
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        self.pad = pad
        if c_in > c_out:
            self.down_sample_conv_weights = wb.get_w(name="down_sample_conv_weights", shape=[1,1,c_in,c_out], dtype=tf.float64, trainable=True)
        if act_func == "GLU":
            c_o = 2*c_out
        else:
            c_o = c_out
        self.dense_weights = wb.get_w(name="dense_weights", shape=[Kt, 1, c_in, c_o], dtype=tf.float64, trainable=True)
        self.dense_bias =  wb.get_b(name="dense_bias", shape=[c_o], dtype=tf.float64, trainable=True)

    @tf.function
    def __call__(self, x):
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


class SpatioConvLayer:
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, wb: WB, graph_kernel, Ks, c_in, c_out):
        self.graph_kernel = tf.Variable(initial_value = graph_kernel, trainable=False)
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        if self.c_in > self.c_out:
            self.down_sample_conv_weights = WB.get_w(name="down_sample_conv_weights", shape=[1,1,self.c_in,self.c_out], dtype=tf.float64, trainable=True)
        self.dense_weights = wb.get_w(name="dense_weights", shape=[self.Ks*self.c_in, self.c_out], dtype=tf.float64, trainable=True)
        self.dense_bias =  wb.get_w(name="dense_bias", shape=[self.c_out], dtype=tf.float64, trainable=True)

    @tf.function
    def __call__(self, x: tf.Tensor):
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


class FullyConLayer:
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param outc: int, output channel size.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    def __init__(self, wb:WB, n, channel, outc):
        self.n = n
        self.channel = channel
        self.outc = outc
        self.dense_weights = wb.get_w(name="dense_weights", shape=[1, 1, self.channel, self.outc], dtype=tf.float64, trainable=True)
        self.dense_bias =  wb.get_w(name="dense_bias", shape=[self.n, self.outc], dtype=tf.float64, trainable=True)

    @tf.function
    def __call__(self, x: tf.Tensor):
        x = tf.cast(x, tf.float64)
        return tf.nn.conv2d(x, self.dense_weights, strides=[1, 1, 1, 1], padding='SAME') + self.dense_bias


class OutputLayer:
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
    def __init__(self, wb: WB, Kt, n, channel, outc=1, act_func="GLU", norm="layer"):
        super().__init__()
        self.Kt = Kt
        self.n = n
        self.outc = outc
        self.act_func = act_func
        self.norm = norm
        self.layer1 = TemporalConvLayer(wb, self.Kt, channel, channel, self.act_func)
        self.layer2 = TemporalConvLayer(wb, 1, channel, channel, self.act_func)
        self.layer3 = FullyConLayer(wb, n, channel, outc)
        if norm == "layer":
            self.normalization = LayerNorm(wb, n, channel)
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')
    
    @tf.function
    def __call__(self, x:tf.Tensor):
        x_i = self.layer1(x)
        if self.norm == "L2":
            x_ln = tf.nn.l2_normalize(x_i, axis=[2,3])
        else:
            x_ln = self.normalization(x_i)
        x_o = self.layer2(x_ln)
        fc = self.layer3(x_o)
        return tf.reshape(fc, shape=[-1, 1, self.n, self.outc])


class STConvBlock:
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
    def __init__(self, wb:WB, graph_kernel, Ks, Kt, channels, act_func='GLU', norm='layer', dropout=0.2, pad='VALID'):
        super().__init__()
        self.norm = norm
        self.rate = dropout
        c_si, c_t, c_oo = channels
        n = graph_kernel.shape[0]
        self.layer1 = TemporalConvLayer(wb,Kt, c_si, c_t, act_func, pad)
        self.layer2 = SpatioConvLayer(wb,graph_kernel, Ks, c_t, c_t)
        self.layer3 = TemporalConvLayer(wb,Kt, c_t, c_oo, act_func, pad)
        if norm == "layer":
            self.normalization = LayerNorm(wb, n, c_oo)
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')

    @tf.function
    def __call__(self, x:tf.Tensor):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        if self.norm == "L2":
            out = tf.nn.l2_normalize(x3, axis=[2,3])
        else:
            out = self.normalization(x3)
        return tf.nn.dropout(out, self.rate)