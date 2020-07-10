import os, sys, random, threading
import cv2, numpy as np


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


def translate(image, x, y):
    # x+: right, x-: left; y+ down, y- up
    matrix = np.array([[1, 0, x], [0, 1, y]], dtype='float32')
    shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def rotate_bound(image, angle):
    """
    :param image:
    :param angle:
    :return:
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    matrix[0, 2] += (nW / 2) - cX
    matrix[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, matrix, (nW, nH))
    # perform the actual rotation and return the image
    return img
