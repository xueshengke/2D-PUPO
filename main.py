from __future__ import print_function
from sources.utils import useGPU
from models import interface
from sources import config_parser, entrance, utils
from datasets import prepare
from skimage import io, transform
import os, sys, json, argparse, cv2


def main(config, train_set, valid_set):
    # create model
    adapter = interface.ModelAdapter(config)
    model = adapter.create_model()
    adapter.serialize_model()

    # initialize handler
    runner = entrance.Runner(model=model, config=config, adapter=adapter)

    if known_args.run == 'train':
        # train model
        history = runner.train(train_set=train_set, valid_set=valid_set)

        runner.validate(test_set=valid_set)

        runner.save_model()

        runner.save_metrics()

    elif known_args.run == 'validate':

        # test model
        scores = runner.validate(test_set=valid_set)

        # runner.save_model()

        runner.save_metrics()

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
    parser.add_argument('--gpu', default='0',
                        type=str, help='Use GPU, e.g., --gpu=0,1,2...')
    parser.add_argument('--run', default='train',
    # parser.add_argument('--run', default='validate',
                        type=str, help='e.g., --run=train, test, prune, or compress')
    parser.add_argument('--dataset', default='MRI_3T',
                        type=str, help='e.g., --model=Cifar-10, Smoke-3, CamVid')
    parser.add_argument('--model', default='VDSR',
                        type=str, help='e.g., --model=ResNet, SmokeNet')
    known_args, unknown_args = parser.parse_known_args()
    if len(unknown_args):
        print('WARNING: unknown arguments: {}. Use "python main.py --help" for details'.format(unknown_args))
    known_args.run = known_args.run.lower()
    known_args.dataset = known_args.dataset.lower()
    known_args.model = known_args.model.lower()

    # use GPU if available, multi-GPU training dest not work now
    sess = useGPU(known_args.gpu)  # '0,1,2,3'

    config_dataset_file = os.path.join('config', 'datasets', known_args.dataset + '.json')
    config_model_file = os.path.join('config', known_args.model, known_args.run + '.json')
    config = config_parser.LoadConfig(known_args, sess)
    config = config.load_dataset_config(config_dataset_file)
    config = config.load_model_config(config_model_file)
    config.serialize()

    # load the dataset
    train_set, valid_set = prepare.load(config)

    # load pre mask
    config.pre_mask = None
    pre_mask_path = ''
    # pre_mask_path = config.poisson_mask_20_path     # load different masks (10%, 20%, 30%, 40%, 50%)
    pre_mask_path = config.synthesize_mask_10_path     # load different masks (10%, 20%, 30%, 40%, 50%)
    if os.path.exists(pre_mask_path):
        print('Load mask from ' + pre_mask_path)
        pre_mask = cv2.imread(pre_mask_path, cv2.IMREAD_GRAYSCALE)
        pre_mask = cv2.resize(pre_mask, config.data_dim[:2], interpolation=cv2.INTER_NEAREST)
        config.pre_mask = utils.normalize(pre_mask.astype('float32'))

    main(config, train_set, valid_set)
