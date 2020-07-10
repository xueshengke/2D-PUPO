from __future__ import print_function
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, PReLU
from keras.layers import AveragePooling2D, Input, Flatten, Add, multiply, Concatenate
from keras.regularizers import l1, l2
from keras.models import Model
from models.attention_modules import insert_attention_module
from models.custom_layers import MaskChannel, PMask2D, IFFT2D, PMask1DH, PMask1DV
import numpy as np, keras, tensorflow as tf


def network(config):
    depth = config.depth
    num_classes = config.num_classes
    input_shape = config.input_data_shape

    # start model definition
    model_arch = {'small': 16, 'medium': 32, 'large': 64}
    num_filters = model_arch[config.arch_size]

    input = Input(shape=input_shape, name='k_input')

    model = PMask2D(name='prob_mask', pre_mask=config.pre_mask)(input)
    # model = PMask1DH(name='prob_mask', pre_mask=config.pre_mask)(input)
    # model = PMask1DV(name='prob_mask', pre_mask=config.pre_mask)(input)

    model = IFFT2D(name='ift')(model)

    assert list(model.shape[1:]) == list(input_shape[:-1]) + [1], 'output of IFFT layer dimension does not match'

    ift_image = model

    for i in range(depth - 1):
        model = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', use_bias=True,
                       kernel_initializer='he_normal', bias_initializer='zeros',
                       kernel_regularizer=l2(1e-6), bias_regularizer=l2(1e-6))(model)
        model = Activation('relu')(model)

    model = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-6))(model)
    residual = model

    output = Add(name='rec')([residual, ift_image])

    model = Model(inputs=input, outputs=[ift_image, output])

    return model
