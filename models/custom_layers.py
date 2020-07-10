from __future__ import print_function
from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import Ones, Zeros, Constant, RandomNormal, RandomUniform
from keras.regularizers import l1, l2
from keras.constraints import MinMaxNorm
from sources.utils import combine_reg, rate_reg, MinMaxLimit, symmetric_reg
import tensorflow as tf, numpy as np, keras


class MaskChannel(Layer):
    def __init__(self, **kwargs):
        super(MaskChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mask = self.add_weight(name='mask', shape=[1, 1, 1, input_shape[-1]], initializer='ones', trainable=False)
        self.output_dim = input_shape
        super(MaskChannel, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        y = x * self.mask
        return y

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(MaskChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PMask1DV(Layer):
    def __init__(self, pre_mask=None, **kwargs):
        self.pre_mask = pre_mask
        super(PMask1DV, self).__init__(**kwargs)

    def build(self, input_shape):
        rate_list = [0.10, 0.20, 0.30, 0.40, 0.50]
        sigma_list = [0.1124, 0.2047, 0.3217, 0.4327, 0.5387]
        cross_list = [0.3070, 0.3789, 0.5712, 0.6825, 0.7049]
        rate = rate_list[4]
        sigma = sigma_list[4] / 1.4142  # / sqrt(2)
        cross = cross_list[4]
        min_prob = max(1.0 * np.exp(-np.square(cross / (np.sqrt(2) * sigma))), 0.01)
        self.prob = self.add_weight(name='probability', trainable=True, shape=[1, 1, input_shape[1], 1],
                                    # initializer=RandomUniform(minval=0., maxval=1.),
                                    initializer=Constant(rate),
                                    constraint=MinMaxLimit(min_value=min_prob, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.03, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.05, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.10, max_value=1.0),
                                    # regularizer=None)
                                    # regularizer=l1(1e-2))
                                    # regularizer=combine_reg)
                                    regularizer=rate_reg(rate))
        self.mask = self.add_weight(name='mask', trainable=False, shape=self.prob.shape, initializer='ones')
        self.output_dim = input_shape
        super(PMask1DV, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):

        if self.pre_mask is not None:
            self.mask = tf.reshape(self.pre_mask, self.prob.shape)
        else:
            self.mask = self.binarize(self.prob)

        y = x * (self.prob + tf.stop_gradient(self.mask - self.prob))
        return y

    def binarize(self, prob):
        prob_vec = tf.reshape(prob, shape=[-1, 1])
        zero_vec = 1. - prob_vec
        prob_mat = tf.concat([zero_vec, prob_vec], axis=1)
        samples = tf.random.categorical(tf.math.log(prob_mat), 1)
        mask = tf.reshape(tf.cast(samples, 'float32'), shape=prob.shape)
        return mask

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(PMask1DV, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PMask1DH(Layer):
    def __init__(self, pre_mask=None, **kwargs):
        self.pre_mask = pre_mask
        super(PMask1DH, self).__init__(**kwargs)

    def build(self, input_shape):
        rate = 0.30
        a = -0.7297 * rate * rate + 0.9053 * rate + 0.8332
        b = 0.0
        c = rate
        # d = -1.37 * rate * rate + 2.368 * rate - 0.0547
        d = -0.7992 * rate * rate + 1.985 * rate + 0.0
        min_prob = a * np.exp(- np.square((d - b) / c))
        self.prob = self.add_weight(name='probability', trainable=True, shape=[1, input_shape[1], 1, 1],
                                    # initializer=RandomUniform(minval=0., maxval=1.),
                                    initializer=Constant(rate),
                                    constraint=MinMaxLimit(min_value=min_prob, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.03, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.05, max_value=1.0),
                                    # constraint=MinMaxLimit(min_value=0.10, max_value=1.0),
                                    # regularizer=None)
                                    # regularizer=l1(1e-2))
                                    # regularizer=combine_reg)
                                    regularizer=rate_reg(rate))
        self.mask = self.add_weight(name='mask', trainable=False, shape=self.prob.shape, initializer='ones')
        self.output_dim = input_shape
        super(PMask1DH, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):

        if self.pre_mask is not None:
            self.mask = tf.reshape(self.pre_mask, self.prob.shape)
        else:
            self.mask = self.binarize(self.prob)

        y = x * (self.prob + tf.stop_gradient(self.mask - self.prob))
        return y

    def binarize(self, prob):
        prob_vec = tf.reshape(prob, shape=[-1, 1])
        zero_vec = 1. - prob_vec
        prob_mat = tf.concat([zero_vec, prob_vec], axis=1)
        samples = tf.random.categorical(tf.math.log(prob_mat), 1)
        mask = tf.reshape(tf.cast(samples, 'float32'), shape=prob.shape)
        return mask

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(PMask1DH, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PMask2D(Layer):
    def __init__(self, pre_mask=None, **kwargs):
        self.pre_mask = pre_mask
        super(PMask2D, self).__init__(**kwargs)

    def build(self, input_shape):
        rate_list = [0.10, 0.20, 0.30, 0.40, 0.50]
        # sigma_list = [0.2373, 0.3359, 0.4114, 0.4751, 0.5311]
        # sigma = sigma_list[4]
        # cross_list = [0.9434, 0.8861, 0.8274, 0.7666, 0.7034]
        # cross = cross_list[4]
        rate = rate_list[0]
        min_prob = rate / np.sqrt(2 * np.pi)
        self.prob = self.add_weight(name='probability', trainable=True, shape=[1, input_shape[1], input_shape[2], 1],
                                    # initializer=RandomUniform(minval=0., maxval=1.),
                                    initializer=Constant(rate),
                                    constraint=MinMaxLimit(min_value=min_prob, max_value=1.0),
                                    # regularizer=None)
                                    # regularizer=l1(1e-2))
                                    regularizer=rate_reg(rate))
        self.mask = self.add_weight(name='mask', trainable=False, shape=self.prob.shape, initializer='ones',
                                    constraint=MinMaxLimit(min_value=0.0, max_value=1.0), regularizer=None)
        self.output_dim = input_shape
        super(PMask2D, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):

        if self.pre_mask is not None:
            self.mask = tf.reshape(self.pre_mask, self.prob.shape)
        else:
            self.mask = self.binarize(self.prob)

        y = x * (self.prob + tf.stop_gradient(self.mask - self.prob))
        return y

    def binarize(self, prob):
        prob_vec = tf.reshape(prob, shape=[-1, 1])
        zero_vec = 1. - prob_vec
        prob_mat = tf.concat([zero_vec, prob_vec], axis=1)
        samples = tf.random.categorical(tf.math.log(prob_mat), 1)
        mask = tf.reshape(tf.cast(samples, 'float32'), shape=prob.shape)
        return mask

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(PMask2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IFFT2D(Layer):
    def __init__(self, **kwargs):
        super(IFFT2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[:-1] + (1,)
        super(IFFT2D, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        c = tf.complex(real, imag)
        c = self.fftshift(c, axes=[1, 2])
        y = K.abs(tf.ifft2d(c))
        y = K.expand_dims(y, axis=-1)
        return y

    def fftshift(self, x, axes=None):
        for axis in axes:
            s = (x.shape[axis] + 1) / 2
            if axis == 0:
                y, z = x[s:, ...], x[:s, ...]
            elif axis == 1:
                y, z = x[:, s:, ...], x[:, :s, ...]
            elif axis == 2:
                y, z = x[:, :, s:, ...], x[:, :, :s, ...]
            elif axis == 3:
                y, z = x[..., s:], x[..., :s]

            x = tf.concat([y, z], axis=axis)
        return x

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(IFFT2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProbMaskChannel(Layer):
    def __init__(self, **kwargs):
        super(ProbMaskChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.prob = self.add_weight(name='prob', trainable=True, shape=[1, 1, 1, self.channel],
                                    initializer='ones', constraint=MinMaxLimit(min_value=0, max_value=1.),
                                    # regularizer=rate_reg)   # gradually increase regularizer force
                                    # regularizer=l01_reg)   # gradually increase regularizer force
                                    # regularizer=l1_max_reg)   # gradually increase regularizer force
                                    # regularizer=neg_xlog)   # gradually increase regularizer force
                                    # regularizer=l1(1e-6))   # gradually increase regularizer force
                                    regularizer=None)  # gradually increase regularizer force
        self.mask = self.add_weight(name='mask', trainable=False, shape=self.prob.shape, initializer='ones')
        # self.output_dim = input_shape
        super(ProbMaskChannel, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        # self.mask = self.binarize(self.prob)
        # self.mask = tf.assign(self.mask, self.binarize(self.prob))
        y = x * (self.prob + tf.stop_gradient(self.mask - self.prob))
        return y

    def binarize(self, prob):
        prob_vec = tf.reshape(prob, shape=[-1, 1])
        zero_vec = 1. - prob_vec
        prob_mat = tf.concat([zero_vec, prob_vec], axis=1)
        samples = tf.random.categorical(tf.math.log(prob_mat), 1)
        mask = tf.reshape(tf.cast(samples, 'float32'), shape=prob.shape)
        return mask

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = { 'output_dim': self.output_dim }
        config = {}
        base_config = super(ProbMaskChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPoolingWithArgmax2D(Layer):

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None, **kwargs):
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
        self.data_format = data_format
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def get_config(self):
        config = {'padding': self.padding, 'pool_size': self.pool_size, 'strides': self.strides,
                  'data_format': self.data_format}
        # config = {}
        base_config = super(MaxPoolingWithArgmax2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), strides=1, padding='same', data_format=None, **kwargs):
        self.up_size = up_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        super(MaxUnpooling2D, self).__init__(**kwargs)

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )

    def get_config(self):
        config = {'up_size': self.up_size, 'strides': self.strides, 'padding': self.padding,
                  'data_format': self.data_format}
        # config = {}
        base_config = super(MaxUnpooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Index(object):
    index = 0

    def __init__(self, id=0):
        self.index = id

    def get(self):
        return self.index

    def set(self, id):
        self.index = id

    def reset(self):
        self.index = 0

    def next(self):
        self.index += 1
        return self


class ScaleChannel(Layer):
    def __init__(self, coeff=1., regularizer=None, **kwargs):
        self.coeff = coeff
        self.regularizer = regularizer
        super(ScaleChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=[1, 1, 1, input_shape[-1]],
                                     initializer=Constant(self.coeff), regularizer=self.regularizer, trainable=True)
        super(ScaleChannel, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        y = x * self.alpha
        return y

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'coeff': self.coeff}
        # config = {}
        base_config = super(ScaleChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OnesZeros(keras.initializers.Initializer):
    """Initializer that generates tensors initialized to [1 x num[0], 0 x num[1]].
    """

    def __init__(self, num=None):
        self.num = num

    def __call__(self, shape, dtype=None):
        if self.num is None:
            self.num = [0, 0]
            from operator import mul
            self.num[0] = reduce(mul, shape, 1) // 2
            self.num[1] = reduce(mul, shape, 1) - self.num[0]
        ones = K.constant(1, shape=(self.num[0],), dtype=dtype)
        zeros = K.constant(0, shape=(self.num[1],), dtype=dtype)
        return K.reshape(K.concatenate([ones, zeros]), shape)


class ReduceChannel(Layer):
    def __init__(self, out_c=None, **kwargs):
        self.out_c = out_c
        super(ReduceChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.in_c = input_shape[-1]
        if self.out_c is None: self.out_c = self.in_c
        self.mask = self.add_weight(name='mask', shape=[1, 1, 1, self.in_c],
                                    initializer=OnesZeros([self.out_c, self.in_c - self.out_c]), trainable=False)
        assert self.in_c >= self.out_c, 'mask channel cannot be larger than input channel'

        super(ReduceChannel, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        mask_vec = K.flatten(self.mask)
        valid_idx = K.flatten(tf.where(K.not_equal(mask_vec, 0)))  # find valid index in mask
        y = tf.gather(x, valid_idx, axis=-1)  # select valid channel from input
        z = y * tf.gather(self.mask, valid_idx, axis=-1)  # multiply valid value in mask
        return z

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_c,)

    def get_config(self):
        config = {'out_c': self.out_c}
        # config = {}
        base_config = super(ReduceChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ExpandChannel(Layer):
    def __init__(self, out_c=None, **kwargs):
        self.out_c = out_c
        super(ExpandChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.in_c = input_shape[-1]
        if self.out_c is None: self.out_c = self.in_c
        self.mask = self.add_weight(name='mask', shape=[1, 1, 1, self.out_c],
                                    initializer=OnesZeros([self.in_c, self.out_c - self.in_c]), trainable=False)
        assert self.out_c >= self.in_c, 'mask channel cannot be smaller than input channel'

        super(ExpandChannel, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x):
        mask_vec = K.flatten(self.mask)
        valid_idx = K.flatten(tf.where(K.not_equal(mask_vec, 0)))  # find valid index in mask
        zero_idx = K.flatten(tf.where(K.equal(mask_vec, 0)))  # find invalid index in mask
        y = x * tf.gather(self.mask, valid_idx, axis=-1)  # multiply valid value in mask
        indices = [tf.cast(valid_idx, 'int32'), tf.cast(zero_idx, 'int32')]
        data = [K.arange(self.in_c), K.constant(-1, shape=(self.out_c - self.in_c,), dtype='int32')]
        expand_idx = tf.dynamic_stitch(indices, data)  # create expand index using valid index
        z = tf.gather(y, expand_idx, axis=-1)  # expand input into new channel
        return z

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_c,)

    def get_config(self):
        config = {'out_c': self.out_c}
        # config = {}
        base_config = super(ExpandChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
