"""
ENet

"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, PReLU, Deconv2D
from keras.layers import AveragePooling2D, Input, Flatten, MaxPool2D, Concatenate, SpatialDropout2D, Add
from keras.regularizers import l1, l2
from keras.models import Model
from models.attention_modules import insert_attention_module

def initial_block(input, is_training=True):
    x1 = Conv2D(filters=13, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=None,
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=l2(1e-6), bias_regularizer=None)(input)
    x2 = MaxPool2D(pool_size=[2, 2], strides=[2, 2])(input)
    x = Concatenate(axis=3)([x1, x2])
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def downsample_block(input, width=16, rate=0.01, is_training=True):
    x1 = Conv2D(filters=width, kernel_size=[2, 2], strides=[2, 2], padding='same', activation=None,
                    use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=l2(1e-6), bias_regularizer=None)(input)
    x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)
    x1 = Conv2D(filters=width, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=None,
                use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x1)
    x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)
    x1 = Conv2D(filters=width*4, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x1)
    x1 = BatchNormalization()(x1)
    x1 = SpatialDropout2D(rate=rate)(x1)

    x2 = MaxPool2D(pool_size=[2, 2], strides=[2, 2])(input)
    x2 = Conv2D(filters=width*4, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                    use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=l2(1e-6), bias_regularizer=None)(x2)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = PReLU()(x)
    return x

def regular_block(input, width=16, rate=0.01, dilate=1, is_training=True):
    x = Conv2D(filters=width, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(input)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    if dilate > 1:
        x = Conv2D(filters=width, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[dilate, dilate],
                   activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)
    else:
        x = Conv2D(filters=width, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=None,
                   use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                   kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=width*4, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(rate=rate)(x)

    y = Add()([x, input])
    y = PReLU()(y)
    return y

def asymmet_block(input, width=16, rate=0.01, kernel=5, is_training=True):
    x = Conv2D(filters=width, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(input)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(filters=width, kernel_size=[kernel, 1], strides=[1, 1], padding='same', activation=None,
               use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)
    x = Conv2D(filters=width, kernel_size=[1, kernel], strides=[1, 1], padding='same', activation=None,
                use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=width*4, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=None,
                use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=l2(1e-6), bias_regularizer=None)(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(rate=rate)(x)

    y = Add()([x, input])
    y = PReLU()(y)
    return y

def enet_v1(input_shape=[None, 32, 32, 3], depth=50, num_classes=10):
    """
    ENet Version 1

    ...

input
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        models (Model): Keras models instance
    """
    # Start models definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    drop_rate = 0.01
    asym_kernel = 5
    dilations = [2, 4, 8, 16]

    ##
    input = Input(shape=input_shape)

    ##
    # x = initial_block(input)
    # x = downsample_block(x, width=num_filters, rate=drop_rate)
    x = downsample_block(input, width=num_filters, rate=drop_rate)

    ##
    for i in range(2):
        x = regular_block(x, width=num_filters, rate=drop_rate)

    ##
    drop_rate = 0.1

    # x = downsample_block(x, width=num_filters, rate=drop_rate)

    ## stage 1
    x = regular_block(x, width=num_filters, rate=drop_rate)
    x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[0])
    x = asymmet_block(x, width=num_filters, rate=drop_rate, kernel=asym_kernel)
    x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[1])
    x = regular_block(x, width=num_filters, rate=drop_rate)
    x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[2])
    x = asymmet_block(x, width=num_filters, rate=drop_rate, kernel=asym_kernel)
    # x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[3])

    num_filters *= 2
    x = downsample_block(x, width=num_filters, rate=drop_rate)

    ## stage 2
    x = regular_block(x, width=num_filters, rate=drop_rate)
    x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[0])
    x = asymmet_block(x, width=num_filters, rate=drop_rate, kernel=asym_kernel)
    x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[1])
    x = regular_block(x, width=num_filters, rate=drop_rate)
    # x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[2])
    # x = asymmet_block(x, width=num_filters, rate=drop_rate, kernel=asym_kernel)
    # x = regular_block(x, width=num_filters, rate=drop_rate, dilate=dilations[3])

    ##
    # x = Deconv2D(filters=19, kernel_size=[1, 1], strides=[1, 1], padding='same', output_padding=None,
    #              dilation_rate=[1, 1], activation=None, use_bias=False, kernel_initializer='glorot_uniform',
    #              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None)(x)
    # output = x

    # classifier
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    output = Dense(units=num_classes, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None)(x)

    # Instantiate models.
    model = Model(inputs=input, outputs=output)
    return model