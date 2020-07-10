from keras.optimizers import Adam, SGD
import sources.loss_func as custom_losses


def initialize(config):
    compile_set = {'resnet': base_compile,
                   'vdsr': vdsr_compile,
                   }
    base_name = config.base_name
    if base_name not in compile_set.keys():
        raise KeyError('No supported compiler for ' + base_name)
    return compile_set[base_name](config)


def base_compile(config):
    losses = 'categorical_crossentropy'
    if config.optimizer == 'SGD':
        optimizer = SGD(lr=config.init_lr, momentum=0.9, nesterov=False)
    elif config.optimizer == 'Adam':
        optimizer = Adam(lr=config.init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0, amsgrad=False)
    metrics = ['accuracy']
    monitor = 'val_accuracy'

    return losses, optimizer, metrics, monitor


def vdsr_compile(config):
    losses = {'ift': 'mean_squared_error', 'rec': custom_losses.cross_domain_mse}
    loss_weights = {'ift': 1., 'rec': 1.}   # with CNN
    # loss_weights = {'ift': 1., 'rec': 0.}  # without CNN
    if config.optimizer == 'SGD':
        optimizer = SGD(lr=config.init_lr, momentum=0.9, nesterov=False)
    elif config.optimizer == 'Adam':
        optimizer = Adam(lr=config.init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0., amsgrad=False)
    metrics = [custom_losses.PSNR, custom_losses.SSIM]
    monitor = 'val_rec_PSNR'

    return losses, loss_weights, optimizer, metrics, monitor
