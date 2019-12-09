from __future__ import print_function
import numpy as np, tensorflow as tf, keras
import os, sys
from datasets import mri_3t


def load(config):
    print('-------- Prepare dataset --------')
    print('Use dataset: ' + config.dataset_name)

    datasets = {
                'mri_3t': mri_3t,
                }
    return datasets[config.dataset_name].create(config)