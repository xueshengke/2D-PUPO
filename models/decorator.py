from __future__ import print_function
import keras, tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, PReLU
from keras.layers import AveragePooling2D, Input, Flatten, Add, multiply, Concatenate, MaxPooling2D, SpatialDropout2D
from keras.models import Model
from keras.regularizers import l1, l2
from keras import activations
from models import attention_modules
from models.custom_layers import MaskChannel, ProbMaskChannel, Index, ScaleChannel, ReduceChannel, ExpandChannel, \
    MaxPoolingWithArgmax2D, MaxUnpooling2D
from sources.utils import get_type


class ModelDecorator(object):
    def __init__(self, org_model, config):
        self.org_model = org_model
        self.config = config
        self.model = None
        self.org_input_nodes = {}
        self.org_output_nodes = {}
        idx_layer = 0

        for org_layer in self.org_model.layers:
            if isinstance(org_layer.input, (list)):
                for node in org_layer.input:
                    node_name = node.name
                    self.__add_org_input_nodes(node_name, get_type(org_layer))
            else:
                node_name = org_layer.input.name
                self.__add_org_input_nodes(node_name, get_type(org_layer))

            if isinstance(org_layer.output, (list)):
                for node in org_layer.output:
                    node_name = node.name
                    self.org_output_nodes[node_name] = [idx_layer, get_type(org_layer)]
                    idx_layer += 1
            else:
                node_name = org_layer.output.name
                self.org_output_nodes[node_name] = [idx_layer, get_type(org_layer)]
                idx_layer += 1

        # sorted(self.org_input_nodes.items(), key=lambda item: item[1])
        # sorted(self.org_output_nodes.items(), key=lambda item: item[1])
        self.input_nodes = []
        self.output_nodes = []
        self.id_conv = 0
        self.id_mask = 0
        self.id_scale = 0
        self.id_reduce = 0
        self.id_expand = 0
        self.set_layer = {'Conv2D': self.set_conv,
                          'Conv2DTranspose': self.set_conv,
                          'BatchNormalization': self.set_batchnorm,
                          'Activation': self.set_activation,
                          'MaxPooling2D': self.set_pool,
                          'AveragePooling2D': self.set_pool,
                          'Add': self.set_add,
                          'Flatten': self.set_flatten,
                          'Dense': self.set_dense,
                          'SpatialDropout2D': self.set_spatial_dropout,
                          'PReLU': self.set_prelu,
                          'ProbMaskChannel': self.set_mask,
                          'MaskChannel': self.set_mask,
                          'MaxPoolingWithArgmax2D': self.set_pool,
                          'MaxUnpooling2D': self.set_unpool,
                          'ScaleChannel': self.set_scale,
                          'ReduceChannel': self.set_reduce_expand,
                          'ExpandChannel': self.set_reduce_expand,
                          }

    def __add_org_input_nodes(self, node_name, layer_type):
        if not node_name in self.org_output_nodes.keys():
            self.org_input_nodes[node_name] = [-1, None, layer_type]
        else:
            self.org_input_nodes[node_name] = self.org_output_nodes[node_name][:2]
            self.org_input_nodes[node_name].append(layer_type)
            self.org_output_nodes[node_name].append(layer_type)

    def __get_org_prev_type(self, node):
        if isinstance(node, (list)):
            prev_nodes = []
            for n in node:
                prev_nodes += self.org_input_nodes[n.name][1:-1]
        else:
            prev_nodes = self.org_input_nodes[node.name][1:-1]
        return prev_nodes

    def __get_org_next_type(self, node):
        if isinstance(node, (list)):
            next_nodes = []
            for n in node:
                next_nodes += self.org_output_nodes[n.name][2:]
        else:
            next_nodes = self.org_output_nodes[node.name][2:]
        return next_nodes

    def __get_input_node(self, org_input_node):
        if isinstance(org_input_node, (list)):
            input_node = []
            for node in org_input_node:
                prev_id = self.org_input_nodes[node.name][0]
                input_node.append(self.output_nodes[prev_id])
        else:
            prev_id = self.org_input_nodes[org_input_node.name][0]
            input_node = self.output_nodes[prev_id]
        return input_node

    def insert_batchnorm(self, x):
        if not self.config.use_bn:  return x
        axis = -1
        momentum = 0.99
        epsilon = 1e-3
        center = True
        scale = True
        beta_initializer = 'zeros'
        gamma_initializer = 'ones'
        moving_mean_initializer = 'zeros'
        moving_variance_initializer = 'ones'
        try:
            beta_reg = l1(self.config.lambda_beta)
        except:
            beta_reg = None
        try:
            gamma_reg = l1(self.config.lambda_gamma)
        except:
            gamma_reg = None
        beta_constraint = None
        gamma_constraint = None
        trainable = True

        batchnorm = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                                       beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                       moving_mean_initializer=moving_mean_initializer,
                                       moving_variance_initializer=moving_variance_initializer,
                                       beta_regularizer=beta_reg, gamma_regularizer=gamma_reg,
                                       beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                                       trainable=trainable)
        if self.config.verbose:
            print('    Insert BatchNormalization(name={})'.format(batchnorm.name))
        x = batchnorm(x)
        return x

    def insert_bn_mask(self, x):
        if self.config.bn_mask:
            if self.config.run == 'auto':
                if self.config.verbose:
                    print('    Insert ProbMaskChannel(name=bn_prob_mask_{})'.format(self.id_mask))
                x = ProbMaskChannel(name='bn_prob_mask_' + str(self.id_mask))(x)
            else:
                if self.config.verbose:
                    print('    Insert MaskChannel(name=bn_mask_{})'.format(self.id_mask))
                x = MaskChannel(name='bn_mask_' + str(self.id_mask))(x)
            self.id_mask += 1
        return x

    def insert_scale(self, x):
        if self.config.use_scale:
            try:
                reg = l1(self.config.lambda_alpha)
            except:
                reg = None
            if self.config.verbose:
                print('    Insert ScaleChannel(regularizer=reg, name=scale_{})'.format(self.id_scale))
            x = ScaleChannel(regularizer=reg, name='scale_' + str(self.id_scale))(x)
            self.id_scale += 1
        return x

    def insert_scale_mask(self, x):
        if self.config.scale_mask:
            if self.config.run == 'auto':
                if self.config.verbose:
                    print('    Insert ProbMaskChannel(name=scale_prob_mask_{})'.format(self.id_mask))
                x = ProbMaskChannel(name='scale_prob_mask_' + str(self.id_mask))(x)
            else:
                if self.config.verbose:
                    print('    Insert MaskChannel(name=scale_mask_{})'.format(self.id_mask))
                x = MaskChannel(name='scale_mask_' + str(self.id_mask))(x)
            self.id_mask += 1
        return x

    def insert_expand(self, x, channel):
        if self.config.final:
            if isinstance(x, (list)):
                y = []
                for t in x:
                    if self.config.verbose:
                        print('    Insert ExpandChannel(out_c={}, name=expand_{})'.format(channel, self.id_expand))
                    t = ExpandChannel(out_c=channel, name='expand_' + str(self.id_expand))(t)
                    self.id_expand += 1
                    y.append(t)
            else:
                if self.config.verbose:
                    print('    Insert ExpandChannel(out_c={}, name=expand_{})'.format(channel, self.id_expand))
                y = [ExpandChannel(out_c=channel, name='expand_' + str(self.id_expand))(x)]
                self.id_expand += 1
            return y
        return x

    def insert_reduce(self, x, channel):
        if self.config.final:
            try:
                out_c = min(channel, self.config.final_reduce_channel[self.id_reduce])
            except:
                out_c = channel
            if self.config.verbose:
                print('    Insert ReduceChannel(out_c={}, name=reduce_{})'.format(out_c, self.id_reduce))
            x = ReduceChannel(out_c=out_c, name='reduce_' + str(self.id_reduce))(x)
            self.id_reduce += 1
        return x

    def insert_attention(self, x):
        if self.config.attention is not '':
            if self.config.verbose:
                print('    Insert attention: ' + self.config.attention)
            x = attention_modules.insert_attention(x, self.config.attention)
        return x

    def set_input(self, layer):
        shape = layer.input_shape
        batch_shape = layer.batch_input_shape
        name = layer.name
        try:
            tensor = layer.tensor
        except:
            tensor = None
        return Input(shape=shape, batch_shape=batch_shape, dtype=layer.dtype, sparse=layer.sparse, tensor=tensor,
                     name=name)

    def set_conv(self, layer, x):
        filters = layer.filters
        if self.config.final:
            try:
                filters = min(filters, self.config.final_conv_channel[self.id_conv])
                self.id_conv += 1
            except:
                pass
        kernel_size = layer.kernel_size
        strides = layer.strides
        padding = layer.padding
        data_format = layer.data_format
        dilation_rate = layer.dilation_rate
        activation = layer.activation
        use_bias = layer.use_bias
        try:
            if not use_bias and self.config.final and self.config.merge_conv_bn:
                use_bias = True
        except:
            pass
        kernel_initializer = layer.kernel_initializer
        bias_initializer = layer.bias_initializer
        kernel_regularizer = layer.kernel_regularizer
        bias_regularizer = layer.bias_regularizer
        activity_regularizer = layer.activity_regularizer
        kernel_constraint = layer.kernel_constraint
        bias_constraint = layer.bias_constraint
        trainable = layer.trainable
        name = layer.name

        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint, trainable=trainable, name=name)
        x = conv(x)
        if self.config.use_bn and 'BatchNormalization' not in self.__get_org_next_type(layer.output):
            x = self.insert_batchnorm(x)
        return x

    def set_batchnorm(self, layer, x):
        if self.config.final and not self.config.use_bn and self.config.merge_conv_bn:
            return x
        axis = layer.axis
        momentum = layer.momentum
        epsilon = layer.epsilon
        center = layer.center
        scale = layer.scale
        beta_initializer = layer.beta_initializer
        gamma_initializer = layer.gamma_initializer
        moving_mean_initializer = layer.moving_mean_initializer
        moving_variance_initializer = layer.moving_variance_initializer
        try:
            beta_regularizer = l1(self.config.lambda_beta)
        except:
            beta_regularizer = layer.beta_regularizer
        try:
            gamma_regularizer = l1(self.config.lambda_gamma)
        except:
            gamma_regularizer = layer.gamma_regularizer
        beta_constraint = layer.beta_constraint
        gamma_constraint = layer.gamma_constraint
        trainable = layer.trainable
        name = layer.name

        batchnorm = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                                       beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                       moving_mean_initializer=moving_mean_initializer,
                                       moving_variance_initializer=moving_variance_initializer,
                                       beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                                       beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                                       trainable=trainable, name=name)
        x = batchnorm(x)
        if self.config.bn_mask and 'MaskChannel' not in self.__get_org_next_type(layer.output) \
                and 'ProbMaskChannel' not in self.__get_org_next_type(layer.output):
            x = self.insert_bn_mask(x)
        return x

    def set_add(self, layer, x):
        if self.config.data_format == 'channels_last':
            channel = layer.output_shape[-1]
        elif self.config.data_format == 'channels_first':
            channel = layer.output_shape[0]

        if self.config.final and 'ExpandChannel' not in self.__get_org_prev_type(layer.input):
            x = self.insert_expand(x, channel)
        x = Add(name=layer.name)(x)
        if self.config.final and 'ReduceChannel' not in self.__get_org_next_type(layer.output):
            x = self.insert_reduce(x, channel)
        if self.config.use_scale and 'ScaleChannel' not in self.__get_org_next_type(layer.output):
            x = self.insert_scale(x)
            x = self.insert_scale_mask(x)
        return x

    def set_activation(self, layer, x):
        activation = layer.output.op.type.lower()

        return Activation(activation=activation, trainable=layer.trainable, name=layer.name)(x)

    def set_prelu(self, layer, x):
        alpha_initializer = layer.alpha_initializer
        alpha_regularizer = layer.alpha_regularizer
        alpha_constraint = layer.alpha_constraint
        shared_axes = layer.shared_axes
        trainable = layer.trainable
        name = layer.name

        prelu = PReLU(alpha_initializer=alpha_initializer, alpha_regularizer=alpha_regularizer,
                     alpha_constraint=alpha_constraint, shared_axes=shared_axes, trainable=trainable, name=name)
        x = prelu(x)
        return x

    def set_pool(self, layer, x):
        pool_size = layer.pool_size
        strides = layer.strides
        padding = layer.padding
        data_format = layer.data_format
        trainable = layer.trainable
        name = layer.name

        if get_type(layer) == 'MaxPooling2D':
            pool = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format,
                             trainable=trainable, name=name)
        elif get_type(layer) == 'AveragePooling2D':
            pool = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format,
                                 trainable=trainable, name=name)
        elif get_type(layer) == 'MaxPoolingWithArgmax2D':
            pool = MaxPoolingWithArgmax2D(pool_size=pool_size, strides=strides, padding=padding,
                                          data_format=data_format, trainable=trainable, name=name)
        else:
            pass
        x = pool(x)
        return x

    def set_unpool(self, layer, x):
        pool_size = layer.up_size
        strides = layer.strides
        padding = layer.padding
        data_format = layer.data_format
        trainable = layer.trainable
        name = layer.name

        if get_type(layer) == 'MaxUnpooling2D':
            unpool = MaxUnpooling2D(up_size=pool_size, strides=strides, padding=padding, data_format=data_format,
                                    trainable=trainable, name=name)
        else:
            pass
        x = unpool(x)
        return x

    def set_flatten(self, layer, x):
        return Flatten(data_format=layer.data_format, trainable=layer.trainable, name=layer.name)(x)

    def set_dense(self, layer, x):
        units = layer.units
        activation = layer.output.op.type.lower()
        use_bias = layer.use_bias
        kernel_initializer = layer.kernel_initializer
        bias_initializer = layer.bias_initializer
        kernel_regularizer = layer.kernel_regularizer
        bias_regularizer = layer.bias_regularizer
        activity_regularizer = layer.activity_regularizer
        kernel_constraint = layer.kernel_constraint
        bias_constraint = layer.bias_constraint
        trainable = layer.trainable
        name = layer.name

        dense = Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, trainable=trainable,
                     name=name)
        x = dense(x)
        return x

    def set_spatial_dropout(self, layer, x):
        if self.config.final: return x
        rate = layer.rate
        data_format = layer.data_format

        return SpatialDropout2D(rate=rate, data_format=data_format, trainable=layer.trainable, name=layer.name)(x)

    def set_mask(self, layer, x):
        trainable = layer.trainable
        name = layer.name

        if get_type(layer) == 'ProbMaskChannel':
            mask = ProbMaskChannel(trainable=trainable, name=name)
        elif get_type(layer) == 'MaskChannel':
            mask = MaskChannel(trainable=trainable, name=name)

        x = mask(x)
        return x

    def set_scale(self, layer, x):
        coeff = layer.coeff
        regularizer = layer.regularizer
        trainable = layer.trainable
        name = layer.name

        scale = ScaleChannel(coeff=coeff, regularizer=regularizer, trainable=trainable, name=name)
        x = scale(x)
        if self.config.scale_mask and 'MaskChannel' not in self.__get_org_next_type(layer.output) \
                and 'ProbMaskChannel' not in self.__get_org_next_type(layer.output):
            x = self.insert_scale_mask(x)
        return x

    def set_reduce_expand(self, layer, x):
        out_c = layer.out_c
        trainable = layer.trainable
        name = layer.name

        if get_type(layer) == 'ReduceChannel':
            func = ReduceChannel(out_c=out_c, trainable=trainable, name=name)
        elif get_type(layer) == 'ExpandChannel':
            func = ExpandChannel(out_c=out_c, trainable=trainable, name=name)

        x = func(x)
        return x

    def append_list(self, the_list, items):
        if isinstance(items, (list)):
            for item in items:
                the_list.append(item)
        else:
            the_list.append(items)

    def run(self):
        print('-------- Decorate begin --------')
        org_layers = self.org_model.layers
        for i in range(len(org_layers)):
            if self.config.verbose:
                print('Decorate #{}: {} ({})'.format(i, org_layers[i].name, get_type(org_layers[i])))
            if get_type(org_layers[i]) in ['InputLayer']:
                node = self.set_input(org_layers[i])
                self.append_list(self.input_nodes, node)
                self.append_list(self.output_nodes, node)

            elif get_type(org_layers[i]) in ['Conv2D', 'Conv2DTranspose']:
                new_layer = self.set_layer[get_type(org_layers[i])]
                org_input_node = org_layers[i].input
                input_node = self.__get_input_node(org_input_node)
                self.append_list(self.input_nodes, input_node)
                output_node = new_layer(org_layers[i], input_node)
                self.append_list(self.output_nodes, output_node)

            elif get_type(org_layers[i]) in ['BatchNormalization']:
                new_layer = self.set_layer[get_type(org_layers[i])]
                org_input_node = org_layers[i].input
                input_node = self.__get_input_node(org_input_node)
                self.append_list(self.input_nodes, input_node)
                output_node = new_layer(org_layers[i], input_node)
                self.append_list(self.output_nodes, output_node)

            elif get_type(org_layers[i]) in ['Activation', 'MaxPooling2D', 'AveragePooling2D', 'Flatten', 'Dense',
                                             'SpatialDropout2D', 'PReLU']:
                new_layer = self.set_layer[get_type(org_layers[i])]
                org_input_node = org_layers[i].input
                input_node = self.__get_input_node(org_input_node)
                self.append_list(self.input_nodes, input_node)
                output_node = new_layer(org_layers[i], input_node)
                self.append_list(self.output_nodes, output_node)

            elif get_type(org_layers[i]) in ['ProbMaskChannel', 'MaskChannel', 'MaxPoolingWithArgmax2D',
                                             'MaxUnpooling2D', 'ScaleChannel', 'ReduceChannel', 'ExpandChannel']:
                new_layer = self.set_layer[get_type(org_layers[i])]
                org_input_node = org_layers[i].input
                input_node = self.__get_input_node(org_input_node)
                self.append_list(self.input_nodes, input_node)
                output_node = new_layer(org_layers[i], input_node)
                self.append_list(self.output_nodes, output_node)

            elif get_type(org_layers[i]) in ['Add']:
                new_layer = self.set_layer[get_type(org_layers[i])]
                org_input_node = org_layers[i].input
                input_node = self.__get_input_node(org_input_node)
                self.append_list(self.input_nodes, input_node)
                output_node = new_layer(org_layers[i], input_node)
                self.append_list(self.output_nodes, output_node)
            else:
                raise TypeError('Unsupported layer type: ' + get_type(org_layers[i]))

        self.model = Model(inputs=self.input_nodes[0], outputs=self.output_nodes[-1])
        self.model.name = self.config.model_name
        print('-------- Decorate end --------')
        return self.model
