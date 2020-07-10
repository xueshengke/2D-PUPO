from __future__ import print_function
import os, sys
from keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow as tf
import numpy as np
from keras import backend as K
import keras.backend.tensorflow_backend as KTF


def useGPU(gpu_id):
    pid = os.getpid()
    if not gpu_id:
        print('---- Use CPU, PID: {} ----'.format(pid))
    else:
        print('---- Use GPU: {}, PID: {} ----'.format(gpu_id, pid))
    # only use limited GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    return sess


def var_step_decay(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): the number of epochs
        init_lr (float): initial learning rate

    # Returns
        lr (float32): current learning rate
    """
    init_lr = 1e-3
    reduce_factor = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    if epoch > 280:
        lr = reduce_factor[6] * init_lr
    elif epoch > 240:
        lr = reduce_factor[5] * init_lr
    elif epoch > 200:
        lr = reduce_factor[4] * init_lr
    elif epoch > 160:
        lr = reduce_factor[3] * init_lr
    elif epoch > 120:
        lr = reduce_factor[2] * init_lr
    elif epoch > 80:
        lr = reduce_factor[1] * init_lr
    elif epoch > 40:
        lr = reduce_factor[0] * init_lr
    else:
        lr = init_lr
    print('Learning rate reduces to: %e' % lr)
    return lr


def fix_step_decay(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): the number of epochs
        init_lr (float): initial learning rate

    # Returns
        lr (float32): current learning rate
    """
    init_lr = 1e-3
    min_lr = 1e-8
    reduce_factor = 0.5
    interval = 40
    lr = init_lr * pow(reduce_factor, epoch // interval)
    lr = max(lr, min_lr)
    print('Learning rate reduces to: %e' % lr)
    return lr


def combine_reg(w, c1=0., c2=1e+1):
    return c1 * l01_reg(w) + c2 * rate_reg(w, rate=0.50)


def l01_reg(w):
    # return (tf.sign(0.5 - w) * (w - 0.5) + 1)
    return - 0.5 * K.mean(K.square(w - 0.5) - 0.25)


def rate_reg(rate=0.5, alpha=10.0):
    def regularizer(x):
        return alpha * 0.5 * K.square(K.mean(K.abs(x)) - rate)

    return regularizer


def symmetric_reg(w, alpha=10.0):
    return alpha * 0.5 * K.mean(K.square(K.mean(w, axis=1) - K.mean(w, axis=2)))


def np_sigmoid(x, a=1.):
    s = 1 / (1 + np.exp(-a * x))
    return s


def tf_sigmoid(x, a=1.):
    s = 1 / (1 + tf.exp(-a * x))
    return s


def binomial(prob):
    prob_vec = np.reshape(prob, [-1, ])
    mask = np.random.binomial(size=prob_vec.size, n=1, p=prob_vec)
    mask = np.reshape(mask, prob.shape)
    return mask


def fftshift(x, axes=None):
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


def normalize(x, min_val=None, max_val=None):
    if min_val is None: min_val = np.min(x)
    if max_val is None: max_val = np.max(x)
    x = 1.0 * (x - min_val) / (max_val - min_val)
    x[x > 1] = 1.
    x[x < 0] = 0.
    return x


def print_model(model, dir='report/', config=None):
    line_width = 120

    print(config.model_name)
    # models.summary(line_length=line_width)
    my_model_summary(model, line_length=line_width)
    print('Saving models summary to ' + os.path.join(dir, config.run + '.txt'))

    # print models.summary to a text file
    std_out = sys.stdout
    f = open(os.path.join(dir, config.run + '.txt'), 'w')
    sys.stdout = f

    print(config.model_name)
    # models.summary(line_length=line_width)
    my_model_summary(model, line_length=line_width)

    sys.stdout.close()
    sys.stdout = std_out


def my_model_summary(model, line_length=None, positions=None):
    """Prints a summary of a model.

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
    """

    if model.__class__.__name__ == 'Sequential':
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers,
        # for logging purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (len(v) == 1 and len(v[0].inbound_layers) > 1):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [.45, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #']
    else:
        line_length = line_length or 98
        positions = positions or [.33, .52, .60, .70, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Mult-Add #', 'Connected to']
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)

    # a var as a list, to call in sub-function (count_mult_add)
    total_mult_add_ops = [0, ]
    mult_cost = [1, ]
    add_cost = [1, ]

    def count_params(weights):
        """Count the total number of scalars composing the weights.

        # Arguments
            weights: An iterable containing the weights on which to compute params

        # Returns
            The total number of scalars composing the weights
        """
        return int(np.sum([K.count_params(p) for p in set(weights)]))

    def count_mult_add(layer):
        cls_name = layer.__class__.__name__
        if cls_name in ['Conv2D']:
            num_ops = np.prod(layer.output_shape[1:])
            mult_add_layer_cost = num_ops * np.prod(layer.kernel_size) * mult_cost[0]
            if layer.use_bias:
                mult_add_layer_cost += num_ops * add_cost[0]

        elif cls_name in ['Dense']:
            num_ops = np.prod(layer.input_shape[1:] + layer.output_shape[1:])
            mult_add_layer_cost = num_ops * mult_cost[0]
            if layer.use_bias:
                mult_add_layer_cost += np.prod(layer.output_shape[1:]) * add_cost[0]

        elif cls_name in ['BatchNormalization']:
            num_ops = np.prod(layer.output_shape[1:])
            mult_add_layer_cost = num_ops * (mult_cost[0] + add_cost[0]) * len(layer.weights) / 2

        elif cls_name in ['Activation']:
            num_ops = np.prod(layer.output_shape[1:])
            mult_add_layer_cost = num_ops * mult_cost[0]

        elif cls_name in ['Add']:
            num_ops = np.prod(layer.output_shape[1:])
            mult_add_layer_cost = num_ops * add_cost[0]

        elif cls_name in ['AveragePooling2D']:
            num_ops = np.prod(layer.input_shape[1:])
            mult_add_layer_cost = num_ops * add_cost[0]

        else:
            mult_add_layer_cost = 0

        total_mult_add_ops[0] += mult_add_layer_cost

        return mult_add_layer_cost

    def print_layer_summary(layer):
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'output_shape error'
        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), count_mult_add(layer)]
        print_row(fields, positions)

    def print_layer_summary_with_connections(layer):
        """Prints a summary for a single layer.

        # Arguments
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                connections.append(inbound_layer + '[' + str(inbound_node_index) + ']['
                                   + str(inbound_tensor_index) + ']')
        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), count_mult_add(layer),
                  first_connection]
        print_row(fields, positions)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', '', connections[i]]
                print_row(fields, positions)

    layers = model.layers
    for i in range(len(layers)):
        if sequential_like:
            print_layer_summary(layers[i])
        else:
            print_layer_summary_with_connections(layers[i])
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)

    model._check_trainable_weights_consistency()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    print('Total params: {:,}; Trainable: {:,}; Non-trainable: {:,}'.format(
        trainable_count + non_trainable_count, trainable_count, non_trainable_count))
    print('Total mult-add operations: {:,}'.format(total_mult_add_ops[0]))
    print('_' * line_length)


class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MinMaxLimit(Constraint):
    """Constrains the weights to between [min_value, max_value].
    """

    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        # y = w * K.cast(K.greater_equal(w, self.min_value), K.floatx())
        # z = y * K.cast(K.less_equal(y, self.max_value), K.floatx())
        # return z
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


def get_type(layer):
    return layer.__class__.__name__


def get_list(model, layer_type):
    return [layer for layer in model.layers if get_type(layer) in layer_type]
