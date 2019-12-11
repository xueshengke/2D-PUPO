from __future__ import print_function
import os, sys, argparse, time
from sources.utils import my_model_summary
from models import info
import keras

default_model_file = '/home/xiaobingt/xueshengke/code/MRIFNNR-keras/results/vdsr-10/train/vdsr-10_loss0.0083_rec_PSNR25.9328.h5'
default_output_dir = os.getcwd()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        type=str,
                        help='path of Keras model')
    parser.add_argument('--output_dir', default=None,
                        type=str,
                        help='output directory to save report file')
    opts = parser.parse_args()
    model_file = opts.model
    output_dir = opts.output_dir

    if model_file is None:
        model_file = default_model_file
    model_path, model_name = os.path.split(model_file)
    report_name = 'report_' + model_name[:-3] + '.txt'
    if output_dir is None:
        output_dir = default_output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Restore pretrained Keras model from ' + model_file)
    model = keras.models.load_model(model_file, custom_objects=info.get_custom_objects())

    my_model_summary(model, line_length=120)

    # print models.summary to a text file
    std_out = sys.stdout
    f = open(os.path.join(output_dir, report_name), 'w')
    sys.stdout = f

    my_model_summary(model, line_length=120)

    sys.stdout.close()
    sys.stdout = std_out

    print('Done')
