from __future__ import print_function
import os, sys, argparse, time

os.sys.path.append('..')
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from models import info
from sources.utils import useGPU, normalize
from sources import config_parser
from datasets import prepare
import sources.loss_func as custom_losses
import tensorflow as tf, numpy as np, keras
import keras.backend.tensorflow_backend as KTF
import pydicom
import cv2
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

# default_model_path = '../results/vdsr-5/train/vdsr-5_loss0.0154_ift_PSNR_Loss25.0649.h5'
# default_model_path = '../results/vdsr-5/train/vdsr-5_loss0.0007_ift_PSNR_Loss34.8355.h5'
# default_model_path = '../results/vdsr-5/train/vdsr-5_loss0.0010_ift_PSNR_Loss32.4259.h5'
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0135_rec_PSNR33.5832.h5'
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0020_rec_PSNR36.0713.h5'
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0020_rec_PSNR36.0713.h5'
# prob mask
# 10%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0010_rec_PSNR35.2568.h5'
# 20%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0005_rec_PSNR38.2398.h5'
# 30%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0003_rec_PSNR40.4101.h5'
# 40%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0002_rec_PSNR41.9716.h5'
# 50%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0003_rec_PSNR42.8021.h5'
pre_mask_path = ''

default_dataset = 'MICCAI_2013'
# 50% trained by MICCAI 2013
default_model_path = '../results/vdsr-10/validate/vdsr-10_loss0.0000_rec_PSNR57.5554.h5'

# Poisson mask
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0015_rec_PSNR33.4249.h5'
# mask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.037_rate0.1019.png' # 10%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0008_rec_PSNR36.4106.h5'
# mask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.024_rate0.1988.png' # 20%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0005_rec_PSNR38.8683.h5'
# mask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.018_rate0.3025.png' # 30%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0003_rec_PSNR40.5833.h5'
# mask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.015_rate0.3916.png' # 40%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0005_rec_PSNR38.0380.h5'
# mask_path = '/home/xiaobingt/xueshengke/dataset/masks/VD_poisson_disc_0.0124_rate0.5032.png' # 50%


def handle_args(config, args):
    args_dict = {}
    for item in args:
        key, value = item.lstrip('-').split('=')
        try:
            value = eval(value)
        except:
            pass
        args_dict[key] = value

    for key, value in args_dict.items():
        if getattr(config, key, None):
            setattr(config, key, value)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='2',
                        type=str, help='Use GPU, e.g., --gpu=0,1,2...')
    parser.add_argument('--dataset', default=None,
                        type=str, help='path of test dataset')
    parser.add_argument('--model_path', default=None,
                        type=str, help='path of test model (.h5 or .pb)')
    known_args, unknown_args = parser.parse_known_args()
    if len(unknown_args):
        print('WARNING: extra arguments: {}. Use "python test.py --help" for details'.format(unknown_args))
    # data_path = known_args.dataset.lower()
    # model_path = known_args.model

    if known_args.dataset is None: known_args.dataset = default_dataset.lower()
    if known_args.model_path is None: known_args.model_path = default_model_path

    # use GPU if available, multi-GPU training dest not work now
    sess = useGPU(known_args.gpu)  # '0,1,2,3'

    # if number is None: number = default_number
    # if data_path is None: data_path = default_data_path
    # if model_path is None: model_path = default_model_path

    config_dataset_file = os.path.join('datasets', known_args.dataset + '.json')
    config = config_parser.LoadConfig(known_args, sess)
    config = config.load_dataset_config(config_dataset_file)

    config = handle_args(config, unknown_args)

    figure_dir = os.path.join(os.getcwd(), os.path.split(config.model_path)[-1][:-3])
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    # # load test images from path, choose number of images to test
    train_set, valid_set = prepare.load(config)

    # test by using keras model (.h5 file)
    if config.model_path.endswith('h5'):
        print('Restore pretrained Keras model from ' + config.model_path)
        model = keras.models.load_model(config.model_path, custom_objects=info.get_custom_objects())

        # load mask
        if os.path.exists(pre_mask_path):
            print('Load mask from ' + pre_mask_path)
            pre_mask = cv2.imread(pre_mask_path)
            pre_mask = cv2.resize(pre_mask, [256, 256], interpolation=cv2.INTER_NEAREST)
            pre_mask = normalize(pre_mask.astype('float32'))
            pre_mask = np.reshape(pre_mask, [1, 256, 256, 1])
            model.get_layer('prob_mask').set_weights([pre_mask, pre_mask])

        time_begin = time.time()
        print('Test started time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if config.use_generator:
            steps = max(1, config.num_valid * 264 / config.batch_size)
            pred = model.predict_generator(valid_set, steps=steps, workers=1, use_multiprocessing=False, verbose=1)
        else:
            pred = model.predict(valid_set[0], batch_size=config.batch_size, verbose=1)

        # pred = model.predict(x_test, batch_size=batch_size, verbose=1)

    # test by using tensorflow model (.pb file)
    elif config.model_path.endswith('pb'):
        print('Restore pretrained Pb model from ' + config.model_path)
        with tf.gfile.FastGFile(config.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            node_list = ''
            for node in graph_def.node:
                node_list += node.name + '\n'
            with open('graph_node_list.txt', 'w') as nlf:
                nlf.write(node_list)

        with tf.Session() as sess:
            input_tensor = graph_def.node[0].name + ':0'
            output_tensor = graph_def.node[-1].name + ':0'
            output = sess.graph.get_tensor_by_name(output_tensor)

            time_begin = time.time()
            print('Test started time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            pred = sess.run(output, {input_tensor: valid_set[0]})

    # pred_cls = np.argmax(pred, axis=1)
    # print('Predict: ' + str(pred))
    # print('Class: ' + str(pred_cls))
    real, imag = np.tanh(valid_set[0][..., 0]), np.tanh(valid_set[0][..., 1])
    prob_mask = np.squeeze(model.get_layer('prob_mask').get_weights()[0])
    ift_out, rec_out = np.squeeze(pred[0]), np.squeeze(pred[1])
    gnd = np.squeeze(valid_set[1])
    ift_psnr, ift_ssim = custom_losses.compute_psnr(ift_out, gnd), custom_losses.compute_ssim(ift_out, gnd)
    rec_psnr, rec_ssim = custom_losses.compute_psnr(rec_out, gnd), custom_losses.compute_ssim(rec_out, gnd)
    print('ift_PSNR: {:.4f}, ift_SSIM: {:.4f}'.format(ift_psnr, ift_ssim))
    print('rec_PSNR: {:.4f}, rec_SSIM: {:.4f}'.format(rec_psnr, rec_ssim))
    for i in range(config.num_valid):
        plt.figure(figsize=(12, 8), dpi=100)
        plt.subplot(3, 3, 1)
        fig_obj = plt.imshow(real[i, ...], cmap=plt.get_cmap('jet'))
        plt.colorbar(fig_obj)
        plt.title('real_' + str(i))
        plt.subplot(3, 3, 2)
        fig_obj = plt.imshow(imag[i, ...], cmap=plt.get_cmap('jet'))
        plt.colorbar(fig_obj)
        plt.title('imag_' + str(i))
        plt.subplot(3, 3, 3)
        fig_obj = plt.imshow(prob_mask, cmap=plt.get_cmap('gray'))
        plt.colorbar(fig_obj)
        plt.title('prob_mask')
        # -------------------
        plt.subplot(3, 3, 4)
        plt.imshow(ift_out[i, ...])
        plt.title('ift_output_' + str(i))
        plt.xlabel('PSNR={:.4f}, SSIM={:.4f}'.format(ift_psnr, ift_ssim))
        plt.subplot(3, 3, 5)
        plt.imshow(rec_out[i, ...])
        plt.title('rec_output_' + str(i))
        plt.xlabel('PSNR={:.4f}, SSIM={:.4f}'.format(rec_psnr, rec_ssim))
        plt.subplot(3, 3, 6)
        plt.imshow(gnd[i, ...])
        plt.title('gnd_' + str(i))
        plt.xlabel('PSNR=--, SSIM=1.0')
        # -------------------
        plt.subplot(3, 3, 7)
        plt.imshow(rec_out[i, ...] - ift_out[i, ...])
        plt.title('|rec - ift|={:.4f}'.format(np.sum(np.abs(rec_out[i, ...] - ift_out[i, ...]))))
        # plt.xlabel(str())
        plt.subplot(3, 3, 8)
        plt.imshow(gnd[i, ...] - ift_out[i, ...])
        plt.title('|gnd - ift|={:.4f}'.format(np.sum(np.abs(gnd[i, ...] - ift_out[i, ...]))))
        # plt.xlabel(str())
        plt.subplot(3, 3, 9)
        plt.imshow(gnd[i, ...] - rec_out[i, ...])
        plt.title('|gnd - rec|={:.4f}'.format(np.sum(np.abs(gnd[i, ...] - rec_out[i, ...]))))
        # plt.xlabel(str())
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'predict_image_{}.png'.format(i)))
        print('Saving figure at ' + os.path.join(figure_dir, 'predict_image_{}.png'.format(i)))
        plt.show(block=False)
        plt.pause(0.01)

    # compute the running time
    time_end = time.time()
    print('Test end time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    elapsed_time = time_end - time_begin
    print('Test total time: {} s / {} samples = {} s/sample'.format(elapsed_time, config.num_valid,
                                                                    1.0 * elapsed_time / config.num_valid))
    print('Done')
