from __future__ import print_function
import os, sys, argparse, time
from skimage import io, transform
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from models import info
from sources.utils import useGPU, normalize
import sources.loss_func as custom_losses
import tensorflow as tf, numpy as np, keras
import keras.backend.tensorflow_backend as KTF
import pydicom
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

# subtract_pixel_mean = False
batch_size = 30
default_number = 10
# mean_pixel_file = '/host/xueshengke/code/channel_prune-keras/test/mean_pixel.npy'
default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/CAO_SHU_ZE'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/JIANG_YU_CHUN'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/JIAO_FANG_AN'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/MA_GUI_HONG'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/QI_FENG_ZHEN'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/SONG_HONG_TAO'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/WANG_AI_LING'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/WANG_CHANG_YUAN'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/YANG_BAO_SHENG'
# default_data_path = '/home/xiaobingt/xueshengke/dataset/3T/ZHAO_GUI_FANG'
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
default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0005_rec_PSNR38.2398.h5'
# 30%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0003_rec_PSNR40.4101.h5'
# 40%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0002_rec_PSNR41.9716.h5'
# 50%
# default_model_path = '../results/vdsr-10/train/vdsr-10_loss0.0003_rec_PSNR42.8021.h5'
mask_path = ''


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


def load_testdata(data_path, num=None):
    # obtain number and filename of test images
    if num is None:
        file_list = os.listdir(data_path)
    else:
        file_list = os.listdir(data_path)[:num]
    image_list = []
    for img_name in file_list:
        image_list.append(os.path.join(data_path, img_name))
    num_image = len(image_list)
    print('Test %d images' % num_image)

    img_size = [256, 256]
    x_test = np.zeros(shape=[num_image] + img_size + [2])
    y_test = np.zeros(shape=[num_image] + img_size + [1])
    for i in range(num_image):
        ima = pydicom.dcmread(image_list[i])
        img = (ima.pixel_array).astype('float32')
        y_test[i, ...] = img[..., np.newaxis]
        print('Read %d / %d: %s' % (i + 1, num_image, image_list[i]))
    y_test = normalize(y_test.astype('float'))

    # # if subtract pixel mean is enabled
    # if subtract_pixel_mean:
    #     if os.path.exists(mean_pixel_file):
    #         print('Load mean pixel from ' + mean_pixel_file)
    #         mean_pixel = np.load(mean_pixel_file)
    #     else:
    #         mean_pixel = np.mean(y_test, axis=0)
    #     y_test -= mean_pixel

    for i in range(num_image):
        # img_f = fft2(np.squeeze(y_train[i, ...]))
        img_f = fftshift(fft2(np.squeeze(y_test[i, ...])))
        x_test[i, ..., 0] = np.real(img_f)
        x_test[i, ..., 1] = np.imag(img_f)

    print('Test_data shape:', x_test.shape)
    return x_test, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='2',
                        type=str,
                        help='Use GPU, e.g., --gpu=0,1,2...')
    parser.add_argument('--dataset', default=None,
                        type=str,
                        help='path of test dataset')
    parser.add_argument('--test_num', default=None,
                        type=int,
                        help='number of images to be tested')
    parser.add_argument('--model', default=None,
                        type=str,
                        help='path of test model')
    opts = parser.parse_args()
    data_path = opts.dataset
    number = opts.test_num
    model_path = opts.model

    useGPU(opts.gpu)  # '0,1,2,3'

    if number is None: number = default_number
    if data_path is None: data_path = default_data_path
    if model_path is None: model_path = default_model_path
    figure_dir = os.path.join(os.getcwd(), os.path.split(model_path)[-1][:-3])
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    # load test images from path, choose number of images to test
    x_test, y_test = load_testdata(data_path, number)

    # test by using keras model (.h5 file)
    if model_path.endswith('h5'):
        print('Restore pretrained Keras model from ' + model_path)
        model = keras.models.load_model(model_path, custom_objects=info.get_custom_objects())

        # load mask
        if os.path.exists(mask_path):
            print('Load mask from ' + mask_path)
            pre_mask = io.imread(mask_path)
            pre_mask = transform.resize(pre_mask, [256, 256])
            pre_mask = normalize(pre_mask.astype('float32'))
            pre_mask = np.reshape(pre_mask, [1, 256, 256, 1])
            model.get_layer('prob_mask').set_weights([pre_mask, pre_mask])

        time_begin = time.time()
        print('Test started time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        pred = model.predict(x_test, batch_size=batch_size, verbose=1)

    # test by using tensorflow model (.pb file)
    elif model_path.endswith('pb'):
        print('Restore pretrained Pb model from ' + model_path)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
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
            pred = sess.run(output, {input_tensor: x_test})

    # pred_cls = np.argmax(pred, axis=1)
    # print('Predict: ' + str(pred))
    # print('Class: ' + str(pred_cls))
    real, imag = np.tanh(x_test[..., 0]), np.tanh(x_test[..., 1])
    prob_mask = np.squeeze(model.get_layer('prob_mask').get_weights()[0])
    ift_out, rec_out = np.squeeze(pred[0]), np.squeeze(pred[1])
    gnd = np.squeeze(y_test)
    ift_psnr, ift_ssim = custom_losses.compute_psnr(ift_out, gnd), custom_losses.compute_ssim(ift_out, gnd)
    rec_psnr, rec_ssim = custom_losses.compute_psnr(rec_out, gnd), custom_losses.compute_ssim(rec_out, gnd)
    print('ift_PSNR: {:.4f}, ift_SSIM: {:.4f}'.format(ift_psnr, ift_ssim))
    print('rec_PSNR: {:.4f}, rec_SSIM: {:.4f}'.format(rec_psnr, rec_ssim))
    for i in range(number):
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

    time_end = time.time()
    print('Test end time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    elapsed_time = time_end - time_begin
    print('Test total time: {} s / {} samples = {} s/sample'.format(elapsed_time, number, 1.0 * elapsed_time / number))
    print('Done')
