from __future__ import print_function
from keras.losses import mean_squared_error, categorical_crossentropy
from keras import backend as K
import tensorflow as tf, numpy as np, keras
import os, sys
import keras.backend.tensorflow_backend as KTF


def l1_norm(x, coeff=1e-4):
    return coeff * K.sum(K.abs(x))


def sparse_ratio(y_true, y_pred, threshold=1e-2):
    # description_vectors = tf.get_collection('cbam_description_vectors')
    description_vectors = tf.get_collection('masked_se_vectors')
    total_num = 0.0
    zero_num = 0.0

    for des_vec in description_vectors:
        mean_des_vec = K.mean(des_vec, axis=[0, 1, 2])
        zero_num += tf.reduce_sum(tf.cast(tf.abs(mean_des_vec) < threshold, 'float32'))
        total_num += tf.cast(tf.size(mean_des_vec), 'float32')

    # return zero_num
    if zero_num * total_num == 0:
        return tf.convert_to_tensor(0.0)
    return tf.convert_to_tensor(zero_num * 1.0 / total_num)


def seblock_L1_loss(y_true, y_pred, l1_coeff=1e-6):
    # l2_loss = mean_squared_error(y_true, y_pred)
    l2_loss = categorical_crossentropy(y_true, y_pred)
    # cbam_description_vectors = tf.get_collection('cbam_description_vectors')
    description_vectors = tf.get_collection('masked_se_vectors')
    l1_loss = 0
    for des_vec in description_vectors:
        l1_loss += l1_norm(des_vec, l1_coeff)
    return l1_loss + l2_loss


def bn_sparse_crossentropy_loss(y_true, y_pred, l1_coeff=0.0):
    # l2_loss = mean_squared_error(y_true, y_pred)
    l2_loss = categorical_crossentropy(y_true, y_pred)

    unmask_bn_features = tf.get_collection('unmask_bn_features')
    masked_bn_features = tf.get_collection('masked_bn_features')

    epsilon = 1e-20
    l1_loss = 0
    for unmask_bn_feature in unmask_bn_features:
        sparse_vec = K.sqrt(K.sum(K.square(unmask_bn_feature), axis=[0, 1, 2]) + epsilon)
        l1_loss += l1_norm(sparse_vec, l1_coeff)

    return l1_loss + l2_loss


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred, max_pixel=1.0):
    # assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def compute_psnr(y_true, y_pred, max_pixel=1.0):
    assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % \
                                         (str(y_true.shape), str(y_pred.shape))
    return 10.0 * np.log10((max_pixel ** 2) / (np.mean(np.square(y_pred - y_true))))


def SSIM(y_true, y_pred):
    # assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = K.square(0.01 * 7)
    c2 = K.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


def compute_ssim(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % \
                                         (str(y_true.shape), str(y_pred.shape))
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


def cross_domain_mse(y_true, y_pred):
    y_loss = tf.losses.mean_squared_error(y_true, y_pred)
    f_rep = tf.fft2d(tf.cast(y_true, 'complex64'))
    f_true = tf.concat([tf.real(f_rep), tf.imag(f_rep)], axis=-1)
    f_rep = tf.fft2d(tf.cast(y_pred, 'complex64'))
    f_pred = tf.concat([tf.real(f_rep), tf.imag(f_rep)], axis=-1)
    f_loss = tf.reduce_mean(tf.square(tf.math.tanh(f_true) - tf.math.tanh(f_pred)))

    return 1.0 * y_loss + 0.0 * f_loss
    # return 1.0 * y_loss + 1.0 * f_loss
    # return 0.1 * y_loss + 1.0 * f_loss
    # return 1.0 * y_loss + 0.1 * f_loss
    # return 1.0 * y_loss + 0.5 * f_loss
    # return 0.5 * y_loss + 1.0 * f_loss
