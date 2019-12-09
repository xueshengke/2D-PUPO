from __future__ import print_function
from sources.utils import get_type, get_list, binomial
import keras.backend as K
import os, sys, time, csv
import tensorflow as tf
import numpy as np
import keras


class BatchNormSparseRate(keras.callbacks.Callback):

    def __init__(self, config, **kwargs):
        self.config = config
        super(BatchNormSparseRate, self).__init__(**kwargs)

    def on_train_begin(self, logs={}):
        self.gamma_sparse_rate = []
        self.beta_sparse_rate = []
        self.threshold = 1e-2
        self.gamma_total_num = 0
        self.beta_total_num = 0
        self.layer_list = get_list(self.model, ['BatchNormalization'])
        for layer in self.layer_list:
            weights = layer.get_weights()
            gamma, beta = weights[0], weights[1]
            self.gamma_total_num += len(gamma)
            self.beta_total_num += len(beta)

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % 10 != 0: return
        gamma_zero_num = 0.0
        beta_zero_num = 0.0
        for layer in self.layer_list:
            weights = layer.get_weights()
            gamma, beta = weights[0], weights[1]
            gamma_zero_num += np.sum(np.abs(gamma) < self.threshold)
            beta_zero_num += np.sum(np.abs(beta) < self.threshold)

        gamma_val = gamma_zero_num * 1.0 / self.gamma_total_num
        beta_val = beta_zero_num * 1.0 / self.beta_total_num

        if self.config.verbose:
            print(' - gamma_sparse_rate: %.4f - beta_sparse_rate: %.4f' % (gamma_val, beta_val))

    def on_epoch_end(self, epoch, logs={}):
        gamma_zero_num = 0.0
        beta_zero_num = 0.0
        for layer in self.layer_list:
            weights = layer.get_weights()
            gamma, beta = weights[0], weights[1]
            gamma_zero_num += np.sum(np.abs(gamma) < self.threshold)
            beta_zero_num += np.sum(np.abs(beta) < self.threshold)

        gamma_val = gamma_zero_num * 1.0 / self.gamma_total_num
        beta_val = beta_zero_num * 1.0 / self.beta_total_num

        logs['gamma_sparse'] = gamma_val
        logs['beta_sparse'] = beta_val
        # if self.config.verbose:
        #     print('Epoch %d: gamma_sparse_rate: %.4f - beta_sparse_rate: %.4f' % (epoch+1, gamma_val, beta_val))


class LogWriter(keras.callbacks.Callback):
    def __init__(self, filename='logs.txt', separator=',', **kwargs):
        self.filename = filename
        self.separator = separator
        self.time_train_begin = 0
        self.time_train_end = 0
        self.time_epoch_begin = 0
        self.time_epoch_end = 0
        super(LogWriter, self).__init__(**kwargs)

    def on_train_begin(self, logs=None):
        self.time_train_begin = time.time()
        print('Training started time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def on_train_end(self, logs=None):
        self.time_train_end = time.time()
        elapsed_time = self.time_train_end - self.time_train_begin
        print('Training total time: ' + self.time_format_convert(elapsed_time))

    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch_begin = time.time()
        print('Epoch started time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def on_epoch_end(self, epoch, logs=None):
        self.time_epoch_end = time.time()
        elapsed_time = self.time_epoch_end - self.time_epoch_begin
        print('Epoch elapsed time: ' + self.time_format_convert(elapsed_time))

        lr = K.get_value(self.model.optimizer.lr)
        logs['real_lr'] = lr
        keys = ['time', 'epoch', 'real_lr']
        values = [time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), epoch, lr]
        info = 'Epoch: %d - learning_rate: %g' % (epoch, lr)
        for key, value in sorted(logs.items()):
            info += ' - %s: %g' % (key, value)
            keys.append(key)
            values.append(value)
        print(info)

        with open(self.filename, 'a+') as csvfile:
            f = csv.writer(csvfile)
            if epoch == 0:
                f.writerow(keys)
            f.writerow(values)

        print('------------------------------------------------------------')

    def time_format_convert(self, t):

        milsec = int((t - int(t)) * 1000)
        t = int(t)
        sec = t % 60
        t = t // 60
        min = t % 60
        t = t // 60
        hour = t % 24
        t = t // 24
        day = t % 365
        t = t // 365
        assert t == 0, 'time is too large to convert'
        str_time = '%d day %d hour %d min %d s %d ms' % (day, hour, min, sec, milsec)
        return str_time


class UpdateProbMask(keras.callbacks.Callback):
    def __init__(self, config, **kwargs):
        self.config = config
        super(UpdateProbMask, self).__init__(**kwargs)

    def on_train_begin(self, logs={}):
        self.pmask_layer = self.model.get_layer('prob_mask')
        # self.conv1_layer = self.model.get_layer('conv2d_1')
        prob, mask = self.pmask_layer.get_weights()
        new_mask = binomial(prob)
        self.pmask_layer.set_weights([prob, new_mask])
        self.mean_prob = np.mean(prob)
        self.mask_rate = 1.0 * np.count_nonzero(new_mask) / new_mask.size

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch > 0 and epoch % 5 == 0:
            prob, mask = self.pmask_layer.get_weights()
            new_mask = binomial(prob)
            self.pmask_layer.set_weights([prob, new_mask])
            self.mean_prob = np.mean(prob)
            self.mask_rate = 1.0 * np.count_nonzero(new_mask) / new_mask.size
        logs['mean_prob'] = self.mean_prob
        logs['mask_rate'] = self.mask_rate

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % 10 != 0: return
        if self.config.verbose:
            print(' - mean_prob: {:.4f} - mask_rate: {:.4f}'.format(self.mean_prob, self.mask_rate))

        # obtain variables
        pmask_input = self.pmask_layer.input
        pmask_output = self.pmask_layer.output
        prob = self.pmask_layer.weights[0]

        # compute gradient of loss w.r.t. pmask
        grad_output = K.gradients(self.model.total_loss, [pmask_input])[0]
        grad_mask = grad_output * pmask_input
        grad_mask = K.sum(grad_mask, axis=-1, keepdims=True)
        grad_mask = K.mean(grad_mask, axis=0, keepdims=True)

        # update probability using the approximate gradient
        new_prob = prob - self.model.optimizer.lr * grad_mask
        if prob.constraint is not None:
            prob.constraint(new_prob)
        # K.update(prob, new_prob)
        # self.pmask_layer.set_weights(K.eval(new_prob))
        # prob.assign(new_prob)
        self.pmask_layer.prob.assign_sub(self.model.optimizer.lr * grad_mask)

        kernel = self.conv1_layer.weights[0]
        grad_kernel = K.gradients(self.model.total_loss, [kernel])[0]
        new_kernel = kernel - self.model.optimizer.lr * grad_kernel
        if kernel.constraint is not None:
            kernel.constraint(new_kernel)
        kernel.assign(new_kernel)
        kernel = K.update(kernel, new_kernel)
        self.conv1_layer.kernel.assign_sub(self.model.optimizer.lr * grad_kernel)

        self.model.optimizer.apply_gradients(grads_and_vars=zip(grad_kernel, kernel))
