from __future__ import print_function
from keras.models import load_model, model_from_json, model_from_yaml
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau, \
                            TerminateOnNaN
from sources import compiler
from models.custom_callbacks import BatchNormSparseRate, LogWriter, UpdateProbMask
from keras.utils import multi_gpu_model
import os, sys, time, csv
import sources.utils as util
import decorator, info


class ModelAdapter(object):

    def __init__(self, config):
        self.config = config
        self.model = None

    def create_model(self):
        if self.config.restore_model:
            print('Restore pretrained model from ' + self.config.pre_model_path)

            custom_objects = info.get_custom_objects(self.config)
            if self.config.pre_model_path.endswith('h5'):
                model = load_model(self.config.pre_model_path, custom_objects=custom_objects)
            elif self.config.pre_model_path.endswith('json'):
                with open(self.config.pre_model_path, 'r') as f:
                    model_json = f.read()
                model = model_from_json(model_json, custom_objects=custom_objects)
            elif self.config.pre_model_path.endswith('yaml'):
                with open(self.config.pre_model_path, 'r') as f:
                    model_yaml = f.read()
                model = model_from_yaml(model_yaml, custom_objects=custom_objects)

            if self.config.decorate:
                decorate = decorator.ModelDecorator(model, self.config)
                model = decorate.run()
            if self.config.pre_model_path.split('.')[-1] in ['h5']:
                model.load_weights(self.config.pre_model_path, by_name=True)
        else:
            # create model definition
            model = info.get_model_objects(self.config)

            if self.config.decorate:
                decorate = decorator.ModelDecorator(model, self.config)
                model = decorate.run()

            if self.config.restore_weights:
                print('Restore pretrained weights from ' + self.config.pre_model_path)
                # model.load_weights(self.config.pre_model_path, by_name=True)
                model.load_weights(self.config.pre_model_path)

        # if use multiple GPU training
        # parallel_model = multi_gpu_model(model, gpus=4)
        # model = None
        # model = parallel_model

        # define loss, optimizer, and metric
        losses, loss_weights, optimizer, metrics, monitor = compiler.initialize(self.config)
        model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)
        self.config.monitor_metric = monitor
        self.config.record_metrics = ('loss', monitor[4:])
        self.model = model

        return self.model

    def serialize_model(self, model=None, model_report_dir=''):
        if model is None: model = self.model

        # prepare model to report directory
        if model_report_dir == '':
            model_report_dir = os.path.join(self.config.root_dir, 'report', self.config.model_name)
        if not os.path.isdir(model_report_dir):
            os.makedirs(model_report_dir)
        model_json_file = os.path.join(model_report_dir, self.config.run + '.json')
        model_yaml_file = os.path.join(model_report_dir, self.config.run + '.yaml')

        # serialize model to report file
        model_json = model.to_json()
        with open(model_json_file, 'w') as f:
            f.write(model_json)
        print('Saving model summary to ' + str(model_json_file))
        model_yaml = model.to_yaml()
        with open(model_yaml_file, 'w') as f:
            f.write(model_yaml)
        print('Saving model summary to ' + str(model_yaml_file))
        util.print_model(model, model_report_dir, self.config)

    def initialize_callbacks(self, ckpt_save_path='', log_dir=''):

        # prepare model checkpoint saving directory
        if ckpt_save_path == '':
            ckpt_save_path = os.path.join(self.config.root_dir, 'checkpoints', self.config.model_name, self.config.run)
        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        epoch_file = 'epoch{epoch:03d}_%s{%s:02.4f}.h5' % (self.config.monitor_metric[4:], self.config.monitor_metric)
        ckp_file = os.path.join(ckpt_save_path, epoch_file)

        # prepare callbacks for model saving and for learning rate adjustment
        checkpoint = ModelCheckpoint(filepath=ckp_file, monitor=self.config.monitor_metric, verbose=1,
                                     save_best_only=True, mode='max', save_weights_only=self.config.save_weights_only,
                                     period=self.config.save_ckpt_period)

        # prepare tensorboard directory
        if log_dir == '':
            log_dir = os.path.join(self.config.root_dir, 'logs', self.config.model_name, self.config.run)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=True,
                                  write_images=True)

        nan_stop = TerminateOnNaN()

        early_stop = EarlyStopping(monitor=self.config.monitor_metric, patience=self.config.stop_patience, verbose=1,
                                   min_delta=1e-4, mode='max', restore_best_weights=True)

        # lr_scheduler = LearningRateScheduler(util.var_step_decay)
        lr_scheduler = LearningRateScheduler(util.fix_step_decay)
        lr_reducer = ReduceLROnPlateau(monitor=self.config.monitor_metric, factor=self.config.reduce_factor, cooldown=0,
                                       min_delta=1e-4, mode='max', patience=self.config.reduce_patience,
                                       min_lr=self.config.min_lr)

        # sparse_rate = BatchNormalizationSparseRate()

        self.config.log_file = '%s_%s.txt' % (self.config.model_name, \
                                              time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        log_writer = LogWriter(filename=os.path.join(log_dir, self.config.log_file), separator=',')

        update_prob_mask = UpdateProbMask(self.config)

        # collect all callback functions
        self.callbacks = [checkpoint, tensorboard, nan_stop,
                          early_stop, lr_reducer,
                          # lr_scheduler,
                          # update_prob_mask,
                          log_writer]

        return self.callbacks
