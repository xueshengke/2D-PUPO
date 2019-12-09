from __future__ import print_function
from sources.utils import useGPU, normalize
from models import interface
from sources import config_parser, entrance
import datasets.prepare
from skimage import io, transform
import os, sys, json, argparse


def main(config, train_set, valid_set):
    # create model
    adapter = interface.ModelAdapter(config)
    model = adapter.create_model()
    adapter.serialize_model()

    # initialize handler
    runner = entrance.Runner(model=model, config=config, adapter=adapter)

    if opts.run == 'train':
        # train model
        history = runner.train(train_set=train_set, valid_set=valid_set)

        runner.validate(test_set=valid_set)

        runner.save_model()

        runner.save_metrics()

    elif opts.run == 'validate':

        # test model
        scores = runner.validate(test_set=valid_set)

    # elif opts.run == 'prune':
    #
    #     # prune model
    #     runner.prune(train_set=train_set, valid_set=valid_set)
    #
    # elif opts.run == 'compress':
    #
    #     # compress model
    #     final_model = runner.compress()
    #
    #     scores = runner.validate(model=final_model, test_set=valid_set)
    #
    #     runner.save_model(model=final_model)
    #
    # elif opts.run == 'auto':
    #
    #     # pretrain model
    #     if not (config.restore_model or config.restore_weights):
    #         history = runner.train(train_set=train_set, valid_set=valid_set)
    #
    #         runner.validate(test_set=valid_set)
    #
    #         runner.save_model()
    #
    #         runner.save_metrics()
    #
    #     # prune model
    #
    #     runner.auto_prune(train_set=train_set, valid_set=valid_set)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',   default='1',     type=str,  help='Use GPU, e.g., --gpu=0,1,2...')
    parser.add_argument('--run',   default='Train', type=str,  help='e.g., --run=train, test, prune, or compress')
    parser.add_argument('--data', default='MRI_3T', type=str, help='e.g., --model=Cifar-10, Smoke-3, CamVid')
    parser.add_argument('--model', default='VDSR',  type=str,  help='e.g., --model=ResNet, SmokeNet')
    opts = parser.parse_args()
    opts.run = opts.run.lower()
    opts.data = opts.data.lower()
    opts.model = opts.model.lower()

    # use GPU if available, multi-GPU training dest not work now
    useGPU(opts.gpu)  # '0,1,2,3'

    config_data_file = os.path.join('config', 'datasets', opts.data + '.json')
    config_model_file = os.path.join('config', opts.model, opts.run + '.json')
    config = config_parser.LoadConfig(opts)
    config = config.load_data_config(config_data_file)
    config = config.load_model_config(config_model_file)

    # load the dataset
    train_set, valid_set = datasets.prepare.load(config)

    # load pre mask
    config.pre_mask = None
    # premask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.037_rate0.1019.png'
    # premask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.024_rate0.1988.png'
    # premask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.018_rate0.3025.png'
    # premask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.015_rate0.3916.png'
    # premask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.0124_rate0.5032.png'
    # if os.path.exists(premask_path):
    #     print('Load mask from ' + premask_path)
    #     pre_mask = io.imread(premask_path)
    #     pre_mask = transform.resize(pre_mask, config.img_size)
    #     config.pre_mask = normalize(pre_mask.astype('float32'))

    main(config, train_set, valid_set)
