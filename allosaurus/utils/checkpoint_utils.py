# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Union
import collections
import logging
import os
import re
import traceback
import shutil

import torch
from torch.serialization import default_restore_location
from pathlib import Path
from allosaurus.config import allosaurus_config
from allosaurus.bin.download_model import download_model

#from fairseq.models import FairseqEncoder, FairseqDecoder

def resolve_model_name(model_name='latest', checkpoint=None):

    public_models = {
        'latest': '23020401',
        '23020401': '23020401',
        '23042401_finetune_transformer_block': '23042401_finetune_transformer_block',
        '23060201_eng': '23060201_eng',
        '23060203_jpn': '23060203_jpn'
    }

    # public version
    if allosaurus_config.repo_name == 'allosaurus':

        model_name = public_models[model_name]

        assert model_name in public_models, f"{model_name} is not available"

        download_model(model_name)

        checkpoint = find_topk_models(model_name, topk=1)[0]

        return model_name, checkpoint

    # private version
    if checkpoint is not None:
        assert Path(checkpoint).exists(), f"{checkpoint} does not exist!!"
        model_name = Path(checkpoint).parent.name
        return model_name, checkpoint
    else:
        assert model_name is not None

        if model_name in public_models:
            model_name = public_models[model_name]

        checkpoint = find_topk_models(model_name, topk=1)[0]

        return model_name, checkpoint


def torch_save(model, path):
    """Save torch arch states.

    Args:
        path (str): Model path to be saved.
        model (torch.nn.Module): Torch arch.

    """
    path = str(path)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(model, path):
    """Load torch arch states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch arch.

    """
    model_state_dict = torch.load(str(path), map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():

        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k

        new_state_dict[name] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    del model_state_dict, new_state_dict


def find_topk_models(exp_name, topk=1):

    exp_dir = allosaurus_config.data_path / 'model' / exp_name

    if (exp_dir / 'model.pt').exists() and topk==1:
        return [exp_dir / 'model.pt']

    model_lst = []
    for model_path in exp_dir.glob('*.pt'):
        perf = model_path.stem.split('_')[1]
        model_lst.append((float(perf), model_path))

    model_lst.sort()
    topk_models = [model[1] for model in model_lst[:topk]]
    return topk_models


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    from pyspeech.ml.torch import distributed_utils, meters

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if args.no_save or not distributed_utils.is_master(args):
        return

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or is_better(val_loss, save_checkpoint.best))
    )
    checkpoint_conds['checkpoint_last.pt'] = not args.no_last_checkpoints

    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            shutil.copyfile(checkpoints[0], cp)

        write_timer.stop()
        print('| saved checkpoint {} (epoch {} @ {} updates) (writing took {} seconds)'.format(
            checkpoints[0], epoch, updates, write_timer.sum))

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r'checkpoint(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, data_selector=None):
    """Load a checkpoint and restore the training iterator."""
    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.restore_file == 'checkpoint_last.pt':
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint_last.pt')
    else:
        checkpoint_path = args.restore_file

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )

    if (
        extra_state is not None
        and 'best' in extra_state
        and not args.reset_optimizer
        and not args.reset_meters
    ):
        save_checkpoint.best = extra_state['best']

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state['train_iterator']
        epoch_itr = trainer.get_train_iterator(epoch=itr_state['epoch'], load_dataset=True, data_selector=data_selector)
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(epoch=0, load_dataset=True, data_selector=data_selector)

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
        # if path manager not found, continue with local file.
    state = torch.load(path, map_location=lambda s, l: default_restore_location(s, 'cpu'),)

    args = state['args']
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble(filenames, arg_overrides=None, task=None):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override arch args that
            were used during arch training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    ensemble, args, _task = load_model_ensemble_and_task(filenames, arg_overrides, task)
    return ensemble, args


def load_model_ensemble_and_task(filenames, arg_overrides=None, task=None):
    from fairseq import tasks

    ensemble = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = load_checkpoint_to_cpu(filename, arg_overrides)

        args = state['args']
        if task is None:
            task = tasks.setup_task(args)

        # build arch for ensemble
        model = task.build_model(args)
        model.load_state_dict(state['arch'], strict=True)
        ensemble.append(model)
    return ensemble, args, task


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(
    filename, args, model_state_dict, criterion, optimizer, lr_scheduler,
    num_updates, optim_history=None, extra_state=None,
):
    from fairseq import utils
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'arch': model_state_dict if model_state_dict else {},
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'extra_state': extra_state,
    }
    if utils.has_parameters(criterion):
        state_dict['criterion'] = criterion.state_dict()
    if not args.no_save_optimizer_state:
        state_dict['last_optimizer_state'] = convert_state_dict_type(optimizer.state_dict())

    torch_persistent_save(state_dict, filename)


def _upgrade_state_dict(state):
    """Helper for upgrading old arch checkpoints."""
    from fairseq import models, registry, tasks

    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': 'CrossEntropyCriterion',
                'best_loss': state['best_loss'],
            },
        ]
        state['last_optimizer_state'] = state['optimizer']
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    # reduce optimizer history's memory usage (only keep the last state)
    if 'optimizer' in state['optimizer_history'][-1]:
        state['last_optimizer_state'] = state['optimizer_history'][-1]['optimizer']
        for optim_hist in state['optimizer_history']:
            del optim_hist['optimizer']
    # record the optimizer class name
    if 'optimizer_name' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['optimizer_name'] = 'FairseqNAG'
    # move best_loss into lr_scheduler_state
    if 'lr_scheduler_state' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['lr_scheduler_state'] = {
            'best': state['optimizer_history'][-1]['best_loss'],
        }
        del state['optimizer_history'][-1]['best_loss']
    # keep track of number of updates
    if 'num_updates' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['num_updates'] = 0
    # old arch checkpoints may not have separate source/target positions
    if hasattr(state['args'], 'max_positions') and not hasattr(state['args'], 'max_source_positions'):
        state['args'].max_source_positions = state['args'].max_positions
        state['args'].max_target_positions = state['args'].max_positions
    # use stateful training data iterator
    if 'train_iterator' not in state['extra_state']:
        state['extra_state']['train_iterator'] = {
            'epoch': state['extra_state']['epoch'],
            'iterations_in_epoch': state['extra_state'].get('batch_offset', 0),
        }
    # default to translation task
    if not hasattr(state['args'], 'task'):
        state['args'].task = 'translation'

    # set any missing default values in the task, arch or other registries
    registry.set_defaults(state['args'], tasks.TASK_REGISTRY[state['args'].task])
    registry.set_defaults(state['args'], models.ARCH_MODEL_REGISTRY[state['args'].arch])
    for registry_name, REGISTRY in registry.REGISTRIES.items():
        choice = getattr(state['args'], registry_name, None)
        if choice is not None:
            cls = REGISTRY['registry'][choice]
            registry.set_defaults(state['args'], cls)

    return state


def verify_checkpoint_directory(model_path: Path) -> None:

    if not model_path.exists():
        model_path.mkdir(exist_ok=True, parents=True)

    temp_file_path = model_path / 'dummy'
    try:
        with open(str(temp_file_path), 'w'):
            pass
    except OSError as e:
        (f'| Unable to access checkpoint save directory: {model_path}')
        raise e
    else:
        os.remove(temp_file_path)
