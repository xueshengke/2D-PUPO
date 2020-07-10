from models import custom_layers
from models import vdsr
import sources.loss_func as custom_losses


def get_custom_objects(config=None):
    custom_objects = {'MaskChannel': custom_layers.MaskChannel,
                      'PMask2D': custom_layers.PMask2D,
                      'PMask1DH': custom_layers.PMask1DH,
                      'PMask1DV': custom_layers.PMask1DV,
                      'IFFT2D': custom_layers.IFFT2D,
                      'PSNR': custom_losses.PSNR,
                      'SSIM': custom_losses.SSIM,
                      'cross_domain_mse': custom_losses.cross_domain_mse,
                      }
    return custom_objects


def get_model_objects(config):
    model_objects = {
        'vdsr': vdsr,
    }
    base_name = config.base_name
    if base_name not in model_objects.keys():
        raise KeyError('No supported model for ' + base_name)
    model = model_objects[base_name].network(config)
    return model
