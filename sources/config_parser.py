import os, sys, json


class LoadConfig(object):
    """
        Initialize the LoadConfig class from report config file.
    """

    def __init__(self, args):

        self.gpu = args.gpu
        self.run = args.run
        self.dataset_name = args.data
        self.base_name = args.model
        self.root_dir = os.getcwd()

    def load_data_config(self, config_data_file):
        json_cfg = json.load(open(config_data_file))
        # locals().update(json_cfg)
        tuple_cfg = self.walk(json_cfg)
        for key, value in tuple_cfg:
            if type(value) is unicode: value = value.encode('utf-8')
            setattr(self, key, value)

        if getattr(self, 'data_dim', None):
            self.data_dim = eval(self.data_dim)

        return self

    def load_model_config(self, config_model_file):
        json_cfg = json.load(open(config_model_file))
        # locals().update(json_cfg)
        tuple_cfg = self.walk(json_cfg)
        for key, value in tuple_cfg:
            if type(value) is unicode: value = value.encode('utf-8')
            setattr(self, key, value)

        if getattr(self, 'sparsity', None):
            self.sparsity = eval(self.sparsity)
            self.prune_steps = len(self.sparsity)

        base_name = self.base_name + '-' + str(self.depth)
        if self.attention is not '':
            base_name += '_' + self.attention
        if self.use_bn: base_name += '_bn'
        if self.bn_mask: base_name += '_mask'
        if self.use_scale: base_name += '_scale'
        if self.scale_mask: base_name += '_mask'
        self.model_name = base_name

        return self

    def walk(self, collect):
        for key, value in collect.items():
            if isinstance(value, dict):
                for inkey, invalue in self.walk(value):
                    yield inkey, invalue
            else:
                yield key, value
