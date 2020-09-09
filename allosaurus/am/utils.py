import torch
from collections import OrderedDict
import numpy as np

def torch_load(model, path, device_id, unit_mask=None):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.
        device_id (int): gpu id (-1 indicates cpu only)

    """

    if device_id >= 0:
        model_state_dict = torch.load(str(path),map_location=torch.device(f'cuda:{device_id}'))
    else:
        model_state_dict = torch.load(str(path), map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():

        # no need for lang specific layer in inference model
        if k.startswith('allophone_layer_dict'):
            continue

        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k

        # remap the phone_layer for fine-tuning
        # it will remap phone_layer.weight and phone_layer.bias
        if k.startswith('phone_layer'):
            if unit_mask is not None:
                phone_size = len(unit_mask.target_unit)

                if len(v.shape) == 2:
                    # for weight

                    hidden_size = v.shape[1]
                    new_v = v.new(phone_size, hidden_size)
                else:
                    # for bias

                    assert len(v.shape) == 1, 'phone_layer shape is either 2 or 1'
                    new_v = v.new(phone_size)

                for domain_phone_id, target_phone_id in unit_mask.unit_map.items():
                    new_v[target_phone_id] = v[domain_phone_id]

                v = new_v

        new_state_dict[name] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    if device_id >= 0:
        model = model.cuda(device_id)

    del model_state_dict, new_state_dict


def torch_save(model, path):
    """Save torch model states.

    Args:
        path (str): Model path to be saved.
        model (torch.nn.Module): Torch model.

    """
    path = str(path)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def apply_to_tensor(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

def apply_to_ndarray(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def tensor_to_cuda(sample, device_id=0):

    def _move_to_cuda(tensor):
        return tensor.to(device_id)

    return apply_to_tensor(_move_to_cuda, sample)

def ndarray_to_tensor(sample):

    def _move_to_tensor(dnarray):
        return torch.from_numpy(dnarray)

    return apply_to_ndarray(_move_to_tensor, sample)

def move_to_tensor(sample, device_id=-1):
    """
    move numpy array to torch tensor

    :param sample:
    :param device_id: -1 means cpu, other means gpu device_id
    :return:
    """

    sample = ndarray_to_tensor(sample)

    # move to cuda if device_id provided
    if device_id >= 0:
        sample = tensor_to_cuda(sample, device_id)

    return sample

def move_to_ndarray(sample):

    if sample.is_cuda:
        sample = sample.cpu()

    return sample.data.numpy()
