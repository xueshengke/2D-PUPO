from __future__ import print_function
import numpy as np, tensorflow as tf, keras
import os, sys
from datasets import mri_3t, miccai_2013


def load(config):
    print('-------- Prepare dataset --------')
    print('Use dataset: ' + config.dataset)

    dataset_sets = {
        'mri_3t': mri_3t,
        'miccai_2013': miccai_2013,
    }
    name = config.dataset
    if name not in dataset_sets.keys():
        raise KeyError('No supported dataset preprocess for ' + name)
    return dataset_sets[name].create(config)
