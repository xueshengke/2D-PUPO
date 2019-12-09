import os, sys, json


class LoadConfig(object):
    """
        Initialize the LoadConfig class from report config file.
    """
    def __init__(self, opts):

        self.gpu = opts.gpu
        self.run = opts.run
        self.dataset_name = opts.data
        self.base_name = opts.model

        self.root_dir = os.getcwd()
        self.final_conv_channel = None
        self.final_scale_channel = None
        self.use_generator = False

    def load_data_config(self, config_data_file):
        json_cfg = json.load(open(config_data_file))

        self.dataset_path = str(json_cfg['path'])
        self.num_classes = json_cfg['num_classes']
        data_dim = json_cfg['data_dim']
        self.data_dim = [int(x) for x in data_dim.split(',')]
        self.data_format = json_cfg['data_format']
        self.data_augmentation = json_cfg['data_augmentation']
        self.subtract_pixel_mean = json_cfg['subtract_pixel_mean']

        return self

    def load_model_config(self, config_model_file):
        json_cfg = json.load(open(config_model_file))

        self.depth = json_cfg['model']['depth']
        self.decorate_model = json_cfg['model']['decorate_model']
        self.attention = str(json_cfg['model']['attention'])
        self.bn_mask = json_cfg['model']['bn_mask']
        self.scale_mask = json_cfg['model']['scale_mask']
        self.use_bn = json_cfg['model']['use_bn']
        self.use_scale = json_cfg['model']['use_scale']
        self.final = json_cfg['model']['final']
        self.merge_conv_bn = json_cfg['model']['merge_conv_bn']

        self.batch_size = json_cfg['train']['batch_size']
        self.epochs = json_cfg['train']['epochs']
        self.optimizer = json_cfg['train']['optimizer']
        self.initial_lr = json_cfg['train']['initial_learning_rate']
        self.stop_patience = json_cfg['train']['stop_patience']
        self.reduce_factor = json_cfg['train']['reduce_factor']
        self.reduce_patience = json_cfg['train']['reduce_patience']
        self.min_lr = json_cfg['train']['minimal_learning_rate']
        self.restore_model = json_cfg['train']['restore_model']
        self.restore_weights = json_cfg['train']['restore_weights']
        self.pre_model_path = str(json_cfg['train']['pre_model_path'])
        self.save_metrics = json_cfg['train']['save_metrics']
        self.save_ckpt_period = json_cfg['train']['save_ckpt_period']
        self.save_weights_only = json_cfg['train']['save_weights_only']
        self.lambda_alpha = json_cfg['train']['lambda_alpha']
        self.lambda_gamma = json_cfg['train']['lambda_gamma']
        self.lambda_beta = json_cfg['train']['lambda_beta']
        self.delta_acc = json_cfg['train']['delta_acc']
        self.verbose = json_cfg['train']['verbose']

        sparsity_str = json_cfg['prune']['sparsity']
        self.sparsity = [float(x) for x in sparsity_str.split(',')]
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
