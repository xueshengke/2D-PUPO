from __future__ import print_function
import pydicom
import numpy as np, keras
from skimage import transform
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sources.utils import normalize
from utils import threadsafe_generator, threadsafe_iter, translate, rotate, rotate_bound
import os, sys, random, threading
import cv2


def run_data_argument(data):
    num_data = data.shape[0]
    dim_data = data.shape[1:]
    # pad = 25

    # rotate
    # angles = [45, 90, 135, 180, 225, 270, 315]
    angles = np.random.rand(8) * 360.0
    # print('rotate by', angles)
    gather_data = []
    for k in range(len(angles)):
        rotate_data = np.zeros(shape=data.shape)
        for i in range(num_data):
            shifted = translate(data[i], 0, -25)
            rotated = transform.rotate(shifted, angles[k], resize=True, preserve_range=True)
            rotate_data[i] = transform.resize(rotated, data[i].shape)
            # padded = cv2.copyMakeBorder(shifted, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            # rotated = rotate(padded, angles[k], scale=1.0)
            # rotated = rotate_bound(shifted, angles[k])
            # rotate_data[i] = cv2.resize(rotated, dim_data, interpolation=cv2.INTER_LINEAR)
        gather_data += [rotate_data]
    argument_data = np.concatenate(gather_data, axis=0)
    data = argument_data

    # flip

    # scale
    # scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    # img_size = data.shape[1:]
    # gather_data = []
    # for k in range(len(scales)):
    #     scale_data = np.zeros(shape=data.shape)
    #     for i in range(data.shape[0]):
    #         crop_scale_data = center_crop(data[i], scales[k])
    #         scale_data[i] = transform.resize(crop_scale_data, img_size)
    #     gather_data += [scale_data]
    # argument_data = np.concatenate(gather_data, axis=0)

    return argument_data


@threadsafe_generator
def generate_batch(image_list, batch_size, img_size, use_argument):
    while True:
        for k in range(0, len(image_list), batch_size):
            batch_image_list = image_list[k: k + batch_size]
            batch_label = np.zeros(shape=[len(batch_image_list), img_size[0], img_size[1]], dtype='float32')
            for i in range(len(batch_image_list)):
                ima = pydicom.dcmread(batch_image_list[i])
                img = (ima.pixel_array).astype('float32')
                batch_label[i, ...] = img
            batch_label = normalize(batch_label)

            if use_argument:
                batch_label = run_data_argument(batch_label)

            batch_data = np.zeros(shape=[batch_label.shape[0], img_size[0], img_size[1], 2], dtype='float32')
            for j in range(batch_label.shape[0]):
                img_f = fftshift(fft2(batch_label[j]))
                batch_data[j, ..., 0] = np.real(img_f)
                batch_data[j, ..., 1] = np.imag(img_f)

            batch_label = batch_label[..., np.newaxis]

            yield ({'k_input': batch_data}, {'ift': batch_label, 'rec': batch_label})


def create(config, valid_split=0.05):
    # Load the MRI 3T data
    dataset_path = config.dataset_path
    batch_size = config.batch_size
    num_classes = config.num_classes
    img_size = config.data_dim

    # obtain sub directories
    person_list = []
    for cwd, dirnames, filenames in os.walk(dataset_path):
        for subdir in dirnames:
            person_list.append(os.path.join(cwd, subdir))
    person_list = np.sort(person_list)

    # obtain number and filename of train/valid images
    image_list = []
    num_per_person = []
    for d in person_list:
        num_per_person.append(len(os.listdir(d)))
        sub_list = []
        for img_name in os.listdir(d):
            sub_list.append(os.path.join(d, img_name))
        image_list.append(sub_list)
    num_image = np.sum(num_per_person)

    # shuflle the image list
    for i in range(num_classes):
        np.random.shuffle(image_list[i])

    # compute numbers of each person for valid and train
    num_valid_per_person = [int(x * valid_split) for x in num_per_person]
    num_train_per_person = [num_per_person[i] - num_valid_per_person[i] for i in range(num_classes)]
    num_valid = np.sum(num_valid_per_person)
    num_train = np.sum(num_train_per_person)

    # split train and valid list
    valid_image_list = []
    train_image_list = []
    for i in range(num_classes):
        # valid_image_list.append(image_list[i][: num_valid_per_person[i]])
        # train_image_list.append(image_list[i][num_valid_per_person[i]:])
        valid_image_list += image_list[i][: num_valid_per_person[i]]
        train_image_list += image_list[i][num_valid_per_person[i]:]

    # shuffle the whole image list
    np.random.shuffle(valid_image_list)
    np.random.shuffle(train_image_list)

    if config.use_generator:
        print('[Yes] data generator')
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
            print('[Yes] real-time data augmentation')
        valid_generator = generate_batch(valid_image_list, batch_size, img_size, config.data_augmentation)
        train_generator = generate_batch(train_image_list, batch_size, img_size, config.data_augmentation)

        return train_generator, valid_generator
    else:
        print('[No] data generator')
        # create data and label arrays
        y_valid = np.zeros(shape=[num_valid, img_size[0], img_size[1]])
        y_train = np.zeros(shape=[num_train, img_size[0], img_size[1]])

        # if valid label exists, load it; otherwise, generate one
        valid_label_file = os.path.join(dataset_path, 'valid_label.npy')
        if os.path.exists(valid_label_file):
            print('Load valid label from ' + valid_label_file)
            y_valid = np.load(valid_label_file)
        else:
            # for i in range(num_valid):
            i = 0
            # for img_dir in valid_image_list:
            #     for img_name in img_dir:
            for img_name in valid_image_list:
                print('%d / %d, %s' % (i + 1, num_valid, img_name))
                ima = pydicom.dcmread(img_name)
                img = (ima.pixel_array).astype('float32')
                y_valid[i] = img
                i += 1
            y_valid = normalize(y_valid)
            np.save(valid_label_file, y_valid)

        # if train label exists, load it; otherwise, generate one
        train_label_file = os.path.join(dataset_path, 'train_label.npy')
        if os.path.exists(train_label_file):
            print('Load train label from ' + train_label_file)
            y_train = np.load(train_label_file)
        else:
            # for i in range(num_train):
            i = 0
            # for img_dir in train_image_list:
            #     for img_name in img_dir:
            for img_name in train_image_list:
                print('%d / %d, %s' % (i + 1, num_train, img_name))
                ima = pydicom.dcmread(img_name)
                img = (ima.pixel_array).astype('float32')
                y_train[i] = img
                i += 1
            y_train = normalize(y_train)
            np.save(train_label_file, y_train)

        # if subtract pixel mean is enabled, MR images cannot subtract mean pixel in frequency domain
        # if config.subtract_pixel_mean:
        # x_train_mean = np.mean(x_train, axis=0)
        # x_train -= x_train_mean
        # x_valid -= x_train_mean

        # use data argumentation
        if config.data_augmentation:
            print('[Yes] data augmentation')
            y_valid = run_data_argument(y_valid)
            num_valid = y_valid.shape[0]
            y_train = run_data_argument(y_train)
            num_train = y_train.shape[0]

        x_valid = np.zeros(shape=[num_valid, img_size[0], img_size[1], 2])
        x_train = np.zeros(shape=[num_train, img_size[0], img_size[1], 2])

        # if valid data exists, load it; otherwise, generate one
        valid_data_file = os.path.join(dataset_path, 'valid_data.npy')
        if os.path.exists(valid_data_file):
            print('Load valid data from ' + valid_data_file)
            x_valid = np.load(valid_data_file)
        else:
            for i in range(num_valid):
                img_f = fftshift(fft2(y_valid[i]))
                x_valid[i, ..., 0] = np.real(img_f)
                x_valid[i, ..., 1] = np.imag(img_f)

        # if train data exists, load it; otherwise, generate one
        train_data_file = os.path.join(dataset_path, 'train_data.npy')
        if os.path.exists(train_data_file):
            print('Load train data from ' + train_data_file)
            x_train = np.load(train_data_file)
        else:
            for i in range(num_train):
                img_f = fftshift(fft2(y_train[i]))
                x_train[i, ..., 0] = np.real(img_f)
                x_train[i, ..., 1] = np.imag(img_f)

        # save variables, just load it next time
        # if not os.path.exists(valid_label_file):
        #     np.save(valid_label_file, y_valid)
        # if not os.path.exists(train_label_file):
        #     np.save(train_label_file, y_train)
        # if not os.path.exists(valid_data_file):
        #     np.save(valid_data_file, x_valid)
        # if not os.path.exists(train_data_file):
        #     np.save(train_data_file, x_train)

        y_train = y_train[..., np.newaxis]
        y_valid = y_valid[..., np.newaxis]

        config.num_train = num_train
        config.num_valid = num_valid
        config.batch_data_shape = (batch_size, img_size[0], img_size[1], 2)
        config.input_data_shape = (img_size[0], img_size[1], 2)
        config.batch_label_shape = (batch_size, img_size[0], img_size[1], 1)
        config.input_label_shape = (img_size[0], img_size[1], 1)
        config.data_augmentation = False

        print('Train {} samples, batch shape: {}'.format(num_train, config.batch_data_shape))
        print('Train {} labels, batch shape: {}'.format(num_train, config.batch_label_shape))
        print('Valid {} samples, batch shape: {}'.format(num_valid, config.batch_data_shape))
        print('Valid {} labels, batch shape: {}'.format(num_valid, config.batch_label_shape))

        return (x_train, y_train), (x_valid, y_valid)
