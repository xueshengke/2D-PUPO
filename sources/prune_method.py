from __future__ import print_function
import os, sys, time
import sys
import tensorflow as tf
import numpy as np
from keras import backend as K
import keras.backend.tensorflow_backend as KTF


def prune_batchnorm(model, layer_list, sparsity=0.0):
    if sparsity == 0:
        print('Skip prune! Because sparsity = %s' % sparsity)
        return

    num_layers = len(layer_list)
    gamma_vector = [None] * num_layers
    beta_vector = [None] * num_layers
    mean_vector = [None] * num_layers
    variance_vector = [None] * num_layers
    chn_len_vector = [None] * num_layers

    # obtain variables from batch normalization layer
    for i in range(num_layers):
        weights = layer_list[i].get_weights()
        gamma, beta, mean, variance = weights[0], weights[1], weights[2], weights[3]
        gamma_vector[i] = gamma
        beta_vector[i] = beta
        mean_vector[i] = mean
        variance_vector[i] = variance
        chn_len_vector[i] = len(gamma)

    # concatenate all gamma vectors into a 1D vector
    unsort_candidates = []
    for i in range(num_layers):
        unsort_candidates = np.concatenate((unsort_candidates, np.abs(gamma_vector[i])))

    # sort
    sorted_candidates = np.sort(unsort_candidates)
    total_channel_num = sorted_candidates.shape[0]
    num_to_prune = int(total_channel_num * sparsity)
    threshold = sorted_candidates[num_to_prune]

    # decide 0 / 1 for the 1D vector
    sorted_indices = np.argsort(unsort_candidates)
    unsort_candidates[sorted_indices[ : num_to_prune]] = 0
    unsort_candidates[sorted_indices[num_to_prune : ]] = 1

    print("Overall prune rate: {} / {} = {}, threshold = {}".format(num_to_prune, total_channel_num, float(num_to_prune)/total_channel_num, threshold))

    # create mask vectors
    prune_mask_vectors = [None] * num_layers
    for i in range(num_layers):
        prune_mask_vectors[i], unsort_candidates = np.split(unsort_candidates, [chn_len_vector[i]])
        ## prevent all masks are zero, at least save one channel
        if not any(prune_mask_vectors[i]):
            max_ch = np.argmax(np.abs(gamma_vector[i]))
            prune_mask_vectors[i][max_ch] = 1

    # show prune rate in each batch normalization layer
    for i in range(num_layers):
        num_channel = chn_len_vector[i]
        num_prune_channel = num_channel - np.count_nonzero(prune_mask_vectors[i])
        print("BatchNorm layer {}, sparse rate {} / {} = {}".format(i+1, num_prune_channel, num_channel, float(num_prune_channel)/num_channel))

    # update variables in each batch normalization layer
    for i in range(num_layers):
        layer_list[i].set_weights([ gamma_vector[i] * prune_mask_vectors[i],
                                    beta_vector[i] * prune_mask_vectors[i],
                                    mean_vector[i],
                                    variance_vector[i] ])

    return prune_mask_vectors


def prune_node(des_vecs, sparsity):
    '''
    This function selects some nodes to be pruned according to 'sparsity'
    :param des_vecs: the list of description vectors of nodes
    :param sparsity: target sparsity
    :return: the mask vector, 0 for prune, 1 for keep
    '''

    print('----------------------------------------------')
    print('Start node prune, target sparsity = %.4f' % sparsity)
    print('Description vector list: %d' % len(des_vecs))

    print('Before prune, channels are:')
    num_channel = []
    expand_des_vecs = np.array([])
    for des_vec in des_vecs:
        expand_des_vecs = np.append(expand_des_vecs, des_vec)
        num_channel.append(len(des_vec))
    print(num_channel)
    expand_mask_vecs = np.ones(expand_des_vecs.shape)

    if sparsity == 0:
        print('NO Prune! Sparsity = 0')
    else:
        num_node = len(expand_des_vecs)
        num_prune = int(num_node * sparsity)
        if num_prune >= num_node:
            print('CANNOT prune %d nodes' % num_prune)
            sys.exit(0)

        abs_des_vec = np.abs(expand_des_vecs)
        sort_des_vec = np.sort(abs_des_vec)
        idx_des_vec = np.argsort(abs_des_vec)

        threshold = sort_des_vec[num_prune]
        print('Prune threshold = %.4e' % threshold)
        # for i in range(num_prune):
        #     expand_mask_vecs[idx_des_vec[i]] = 0
        expand_mask_vecs[idx_des_vec[ : num_prune]] = 0
        print('Actual sparsity ratio %d / %d = %.4f' % (num_prune, num_node, num_prune * 1.0 / num_node))

    print('After prune, channels are:')
    mask_vecs = []
    prune_channels = []
    for mask_len in num_channel:
        mask_vec, expand_mask_vecs = np.split(expand_mask_vecs, [mask_len])

        # if all mask_vec is zero, this layer will be useless; is should save 1 channel at least
        des_vec, abs_des_vec = np.split(abs_des_vec, [mask_len])
        if not any(mask_vec):
            mask_vec[np.argmax(des_vec)] = 1
        mask_vecs.append(mask_vec)

        prune_channels.append('%d/%d' % (np.count_nonzero(mask_vec), mask_len))
    print(prune_channels)

    return mask_vecs
