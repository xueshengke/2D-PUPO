from keras.optimizers import Adam, SGD
from sources.loss_func import cross_domain_mse, PSNR, SSIM


def initialize(config):
    base_name = config.base_name

    if base_name == 'resnet':
        losses = 'categorical_crossentropy'
        loss_weights = None
        if config.optimizer == 'SGD':
            optimizer = SGD(lr=config.initial_lr, momentum=0.9, nesterov=False)
        elif config.optimizer == 'Adam':
            optimizer = Adam(lr=config.initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0, amsgrad=False)
        metrics = ['accuracy']
        monitor = 'val_accuracy'

    elif base_name == 'vdsr':
        losses = {'ift': 'mean_squared_error', 'rec': cross_domain_mse}
        loss_weights = {'ift': 1., 'rec': 1.}
        if config.optimizer == 'SGD':
            optimizer = SGD(lr=config.initial_lr, momentum=0.9, nesterov=False)
        elif config.optimizer == 'Adam':
            optimizer = Adam(lr=config.initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0., amsgrad=False)
        metrics = [PSNR, SSIM]
        monitor = 'val_rec_PSNR'

    return losses, loss_weights, optimizer, metrics, monitor
