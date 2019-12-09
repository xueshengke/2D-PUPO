"""
ResNet v1
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""

from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Add, multiply, Concatenate
from keras.regularizers import l1, l2
from keras.models import Model
from models.attention_modules import insert_attention_module
from models.custom_layers import MaskChannel

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', conv_first=True,
                 batch_norm=True, gamma_reg=l1(1e-8), beta_reg=l1(1e-8), bn_mask=True):
    """
    2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
        batch_norm (bool): whether to include batch normalization
        gamma_reg (regularizer): regularizer for gamma of batch normalization layer
        beta_reg (regularizer): regularizer for beta of batch normalization layer
        bn_mask (bool): whether to include mask after batch normalization layer

    # Returns
        x (tensor): output of this layer, tensor as input to the next layer
    """

    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    bn = BatchNormalization(center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                            beta_regularizer=beta_reg, gamma_regularizer=gamma_reg)
    x = inputs
    if conv_first:  # Conv-BatchNorm-Mask-Activation
        x = conv(x)
        if batch_norm:
            x = bn(x)
            tf.add_to_collection('unmask_bn_features', x)
            if bn_mask:
                x = MaskChannel()(x)
            tf.add_to_collection('masked_bn_features', x)
        if activation is not None:
            x = Activation(activation)(x)
    else:   # BatchNorm-Mask-Activation-Conv
        if batch_norm:
            x = bn(x)
            tf.add_to_collection('unmask_bn_features', x)
            if bn_mask:
                x = MaskChannel()(x)
            tf.add_to_collection('masked_bn_features', x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def network(input_shape, config):
    """
    ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
\
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        models (Model): Keras models instance
    """
    depth = config.model_depth
    num_classes = config.num_classes
    bn_mask = config.bn_mask
    attention = config.attention
    lambda_gamma = config.lambda_gamma
    lambda_beta = config.lambda_beta
    is_final = config.is_final
    channel_vector = config.final_channel

    if (depth - 2) % 6 != 0:
        raise ValueError('Depth should be 6n+2 (eg 20, 32, 44, 50 in [a])')

    # start model definition
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    layer_index = 0
    inputs = Input(shape=input_shape)

    channel = min(num_filters, channel_vector[layer_index]) if is_final else num_filters
    layer_index += 1
    x = resnet_layer(inputs=inputs, num_filters=channel, gamma_reg=l1(lambda_gamma), beta_reg=l1(lambda_beta), bn_mask=bn_mask)

    # instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            channel = min(num_filters, channel_vector[layer_index]) if is_final else num_filters
            layer_index += 1
            y = resnet_layer(inputs=x, num_filters=channel, strides=strides,
                             gamma_reg=l1(lambda_gamma), beta_reg=l1(lambda_beta), bn_mask=bn_mask)

            channel = max(num_filters, channel_vector[layer_index]) if is_final else num_filters
            layer_index += 1
            y = resnet_layer(inputs=y, num_filters=channel, activation=None,
                             gamma_reg=l1(lambda_gamma), beta_reg=l1(lambda_beta), bn_mask=bn_mask)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_norm=False, bn_mask=False)

            x = Add()([x, y])
            x = Activation('relu')(x)

            # attention_module
            if attention != '':
                x = insert_attention_module(x, attention)

        # double the channels
        num_filters *= 2

    # add classifier at the end of network
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # instantiate model
    model = Model(inputs=inputs, outputs=outputs)

    return model
