from __future__ import print_function
import pydicom
import numpy as np, keras
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sources.utils import normalize
import os, sys, random, threading
from skimage import transform

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

    def next(self): # python2
        with self.lock:
          return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


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
        valid_image_list.append(image_list[i][: num_valid_per_person[i]])
        train_image_list.append(image_list[i][num_valid_per_person[i] :])

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
        for img_dir in valid_image_list:
            for img_name in img_dir:
                print('%d / %d, %s' % (i+1, num_valid, img_name))
                ima = pydicom.dcmread(img_name)
                img = (ima.pixel_array).astype('float32')
                y_valid[i] = img
                i += 1
        y_valid = normalize(y_valid)

    # if train label exists, load it; otherwise, generate one
    train_label_file = os.path.join(dataset_path, 'train_label.npy')
    if os.path.exists(train_label_file):
        print('Load train label from ' + train_label_file)
        y_train = np.load(train_label_file)
    else:
        # for i in range(num_train):
        i = 0
        for img_dir in train_image_list:
            for img_name in img_dir:
                print('%d / %d, %s' % (i+1, num_train, img_name))
                ima = pydicom.dcmread(img_name)
                img = (ima.pixel_array).astype('float32')
                y_train[i] = img
                i += 1
        y_train = normalize(y_train)

    # if subtract pixel mean is enabled, MR images cannot subtract mean pixel in frequency domain
    # if config.subtract_pixel_mean:
        # x_train_mean = np.mean(x_train, axis=0)
        # x_train -= x_train_mean
        # x_valid -= x_train_mean

    # data argumentation
    if config.data_augmentation:
        angles = [45, 90, 135, 180, 225, 270, 315]
        if y_valid.shape[0] == num_valid:
            org_data = y_valid
            argumen_data = [org_data]
            for k in range(len(angles)):
                rotate_data = np.zeros(shape=org_data.shape)
                for i in range(org_data.shape[0]):
                    rotate_data[i] = transform.rotate(org_data[i], angles[k], resize=False, preserve_range=True)
                argumen_data += [rotate_data]
            y_valid = np.concatenate(argumen_data, axis=0)
        num_valid *= (len(angles) + 1)
        if y_train.shape[0] == num_train:
            org_data = y_train
            argumen_data = [org_data]
            for k in range(len(angles)):
                rotate_data = np.zeros(shape=org_data.shape)
                for i in range(org_data.shape[0]):
                    rotate_data[i] = transform.rotate(org_data[i], angles[k], resize=False, preserve_range=True)
                argumen_data += [rotate_data]
            y_train = np.concatenate(argumen_data, axis=0)
        num_train *= (len(angles) + 1)

    x_valid = np.zeros(shape=[num_valid, img_size[0], img_size[1], 2])
    x_train = np.zeros(shape=[num_train, img_size[0], img_size[1], 2])

    # if valid data exists, load it; otherwise, generate one
    valid_data_file = os.path.join(dataset_path, 'valid_data.npy')
    if os.path.exists(valid_data_file):
        print('Load valid data from ' + valid_data_file)
        x_valid = np.load(valid_data_file)
    else:
        for i in range(num_valid):
            # img_f = fft2(np.squeeze(y_valid[i, ...]))
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
            # img_f = fft2(np.squeeze(y_train[i, ...]))
            img_f = fftshift(fft2(y_train[i]))
            x_train[i, ..., 0] = np.real(img_f)
            x_train[i, ..., 1] = np.imag(img_f)

    # save variables, just load it next time
    if not os.path.exists(valid_label_file):
        np.save(valid_label_file, y_valid)
    if not os.path.exists(train_label_file):
        np.save(train_label_file, y_train)
    if not os.path.exists(valid_data_file):
        np.save(valid_data_file, x_valid)
    if not os.path.exists(train_data_file):
        np.save(train_data_file, x_train)

    y_train = y_train[..., np.newaxis]
    y_valid = y_valid[..., np.newaxis]
    print('Train data shape:',  x_train.shape)
    print('Train label shape:', y_train.shape)
    print('Valid data shape:',   x_valid.shape)
    print('Valid label shape:',  y_valid.shape)

    config.num_train = num_train
    config.num_valid = num_valid
    config.batch_data_shape = (batch_size, img_size[0], img_size[1], 2)
    config.input_data_shape = (img_size[0], img_size[1], 2)
    config.batch_label_shape = (batch_size, img_size[0], img_size[1], 1)
    config.input_label_shape = (img_size[0], img_size[1], 1)
    config.data_augmentation = False

    return (x_train, y_train), (x_valid, y_valid)
