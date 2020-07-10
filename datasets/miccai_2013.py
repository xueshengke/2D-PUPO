from __future__ import print_function
# import pydicom
import numpy as np, keras
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sources.utils import normalize
import os, sys, random, threading
from skimage import transform
import nibabel as nib


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    # def __next__(self):  # python3
    #     with self.lock:
    #         return self.it.__next__()

    def next(self):  # python2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def generate_batch(label, batch_size, img_size, use_argument):
    while True:
        for i in range(0, label.shape[0], batch_size):
            batch_label = label[i: i + batch_size, ...]
            if use_argument:
                batch_label = run_data_argument(batch_label)
            batch_size = batch_label.shape[0]
            batch_data = np.zeros(shape=[batch_size, img_size[0], img_size[1], 2])
            for j in range(batch_size):
                img_f = fftshift(fft2(batch_label[j]))
                batch_data[j, ..., 0] = np.real(img_f)
                batch_data[j, ..., 1] = np.imag(img_f)
            batch_label = batch_label[..., np.newaxis]
            yield ({'k_input': batch_data}, {'ift': batch_label, 'rec': batch_label})


def run_data_argument(data):
    # rotate
    angles = [45, 90, 135, 180, 225, 270, 315]
    # print('rotate by', angles)
    gather_data = [data]
    for k in range(len(angles)):
        rotate_data = np.zeros(shape=data.shape)
        for i in range(data.shape[0]):
            rotate_data[i] = transform.rotate(data[i], angles[k], resize=False, preserve_range=True)
        gather_data += [rotate_data]
    argument_data = np.concatenate(gather_data, axis=0)

    # flip

    # scale

    return argument_data


def create(config):
    # Load the MRI 3T data
    dataset_path = config.dataset_path
    batch_size = config.batch_size
    num_classes = config.num_classes
    img_size = config.data_dim

    # image paths
    train_image_path = os.path.join(dataset_path, 'train')
    valid_image_path = os.path.join(dataset_path, 'valid')

    # obtain image lists
    train_image_list = []
    for cwd, dirnames, filenames in os.walk(train_image_path):
        for filename in filenames:
            if filename.endswith('.nii.gz'):
                train_image_list.append(os.path.join(cwd, filename))

    valid_image_list = []
    for cwd, dirnames, filenames in os.walk(valid_image_path):
        for filename in filenames:
            if filename.endswith('.nii.gz'):
                valid_image_list.append(os.path.join(cwd, filename))

    # shuffle train and valid images
    random.shuffle(train_image_list)
    random.shuffle(valid_image_list)

    # num_train = len(train_image_list)
    # num_valid = len(valid_image_list)
    train_image_list = train_image_list[: num_train]
    valid_image_list = valid_image_list[: num_valid]
    num_train = config.num_train * config.group_size
    num_valid = config.num_valid * config.group_size

    # if valid label exists, load it; otherwise, generate one
    valid_label_file = os.path.join(dataset_path, 'valid_label.npy')
    if os.path.exists(valid_label_file):
        print('Load valid label from ' + valid_label_file)
        y_valid = np.load(valid_label_file)
        if num_valid < y_valid.shape[0]:
            y_valid = y_valid[:num_valid, ...]
    else:
        # for img_dir in valid_image_list:
        y_valid = None
        i = 0
        for img_name in valid_image_list:
            print('%d / %d, %s' % (i + 1, num_valid, img_name))
            img = nib.load(img_name).get_data().swapaxes(0, 2)
            if i > 0:
                y_valid = np.concatenate([y_valid, img], axis=0)
            else:
                y_valid = img
            i += 1
        y_valid = normalize(y_valid.astype('float32'))
        np.save(valid_label_file, y_valid)
    num_valid = y_valid.shape[0]

    # if train label exists, load it; otherwise, generate one
    train_label_file = os.path.join(dataset_path, 'train_label.npy')
    if os.path.exists(train_label_file):
        print('Load train label from ' + train_label_file)
        y_train = np.load(train_label_file)
        if num_train < y_train.shape[0]:
            y_train = y_train[:num_train, ...]
    else:
        # for img_dir in train_image_list:
        y_train = None
        i = 0
        for img_name in train_image_list:
            print('%d / %d, %s' % (i + 1, num_train, img_name))
            img = nib.load(img_name).get_data().swapaxes(0, 2)
            if i > 0:
                y_train = np.concatenate([y_train, img], axis=0)
            else:
                y_train = img
            i += 1
        y_train = normalize(y_train.astype('float32'))
        np.save(train_label_file, y_train)
    num_train = y_train.shape[0]

    # if subtract pixel mean is enabled, MR images cannot subtract mean pixel in frequency domain
    # if config.subtract_pixel_mean:
    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_valid -= x_train_mean

    if config.use_generator:

        config.num_train = num_train
        config.num_valid = num_valid
        config.batch_data_shape = (batch_size, img_size[0], img_size[1], 2)
        config.input_data_shape = (img_size[0], img_size[1], 2)
        config.batch_label_shape = (batch_size, img_size[0], img_size[1], 1)
        config.input_label_shape = (img_size[0], img_size[1], 1)

        print('Train {} samples, batch shape: {}'.format(num_train, config.batch_data_shape))
        print('Train {} labels, batch shape: {}'.format(num_train, config.batch_label_shape))
        print('Valid {} samples, batch shape: {}'.format(num_valid, config.batch_data_shape))
        print('Valid {} labels, batch shape: {}'.format(num_valid, config.batch_label_shape))
        if config.data_augmentation:
            print('[Yes] run-time data augmentation: x8')
        train_generator = generate_batch(y_train, batch_size, img_size, config.data_augmentation)
        valid_generator = generate_batch(y_valid, batch_size, img_size, config.data_augmentation)

        return train_generator, valid_generator
    else:
        # use data argumentation
        if config.data_augmentation:
            print('[Yes] data augmentation')
            y_valid = run_data_argument(y_valid)
            num_valid = y_valid.shape[0]
            y_train = run_data_argument(y_train)
            num_train = y_train.shape[0]

        x_valid = np.zeros(shape=[num_valid, img_size[0], img_size[1], 2])
        x_train = np.zeros(shape=[num_train, img_size[0], img_size[1], 2])

        for i in range(num_valid):
            img_f = fftshift(fft2(y_valid[i]))
            x_valid[i, ..., 0] = np.real(img_f)
            x_valid[i, ..., 1] = np.imag(img_f)

        for i in range(num_train):
            img_f = fftshift(fft2(y_train[i]))
            x_train[i, ..., 0] = np.real(img_f)
            x_train[i, ..., 1] = np.imag(img_f)

        y_train = y_train[..., np.newaxis]
        y_valid = y_valid[..., np.newaxis]

        config.num_train = num_train
        config.num_valid = num_valid
        config.batch_data_shape = (batch_size, img_size[0], img_size[1], 2)
        config.input_data_shape = (img_size[0], img_size[1], 2)
        config.batch_label_shape = (batch_size, img_size[0], img_size[1], 1)
        config.input_label_shape = (img_size[0], img_size[1], 1)

        print('Train {} samples, batch shape: {}'.format(num_train, config.batch_data_shape))
        print('Train {} labels, batch shape: {}'.format(num_train, config.batch_label_shape))
        print('Valid {} samples, batch shape: {}'.format(num_valid, config.batch_data_shape))
        print('Valid {} labels, batch shape: {}'.format(num_valid, config.batch_label_shape))

        return (x_train, y_train), (x_valid, y_valid)
