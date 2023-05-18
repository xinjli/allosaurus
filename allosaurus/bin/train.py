# -*- coding: utf-8 -*-

from allosaurus.am.trainer import Trainer
from allosaurus.am.config import *
from allosaurus.utils.reporter import *
from allosaurus.am.loader import *
from allosaurus.utils.sampler import ConcatBatchSampler
from allosaurus.am.model import read_am
from allosaurus.pm.model import read_pm
from allosaurus.lm.model import read_lm
from allosaurus.utils.distributed_utils import *
import torch.multiprocessing as mp
from torch.utils.data import random_split, ConcatDataset
from allosaurus.utils.checkpoint_utils import *
import os


def read_loaders(config, pm, lm):
    # temporarily single corpus
    # from allosaurus.data.corpus import read_corpus, subset_tr_cv_corpus, subset_tr_cv_corpus_by_speaker

    reporter.info("reading dataset")
    start = time.time()

    corpus_lst = config.corpus_ids
    tr_datasets = []
    cv_datasets = []

    if 'cv_corpus_ids' in config:

        for corpus_id in corpus_lst:
            dataset = read_dataset(corpus_id, pm, lm)
            tr_datasets.append(dataset)
            reporter.success(f"load train dataset: {dataset}", False)

        for corpus_id in config.cv_corpus_ids:
            dataset = read_dataset(corpus_id, pm, lm)
            cv_datasets.append(dataset)
            reporter.success(f"load valid dataset: {dataset}", False)

    else:

        for corpus_id in corpus_lst:
            dataset = read_dataset(corpus_id, pm, lm)
            tr_dataset, cv_dataset = random_split(dataset, [len(dataset) - 50, 50])

            tr_datasets.append(tr_dataset)
            cv_datasets.append(cv_dataset)

            reporter.success(f"load train dataset: {tr_dataset}", False)
            reporter.success(f"load valid dataset: {cv_dataset}", False)

    concat_tr_dataset = ConcatDataset(tr_datasets)
    concat_cv_dataset = ConcatDataset(cv_datasets)

    tr_batch_sampler = ConcatBatchSampler(tr_datasets, batch_size=config.batch_size, shuffle=True)
    cv_batch_sampler = ConcatBatchSampler(cv_datasets, batch_size=config.batch_size, shuffle=False)

    tr_loader = DataLoader(concat_tr_dataset, collate_fn=collate_feats_langs, batch_sampler=tr_batch_sampler)
    cv_loader = DataLoader(concat_cv_dataset, collate_fn=collate_feats_langs, batch_sampler=cv_batch_sampler)

    reporter.success(f"train_am prepare time: {time.time()-start}")

    return tr_loader, cv_loader

def setup_process_group(config):

    if config.ngpu > 1:
        import torch
        import torch.distributed as dist

        device_cnt = torch.cuda.device_count()

        assert device_cnt >= config.ngpu

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        dist.init_process_group(world_size=config.world_size, backend='nccl', rank=config.rank)
        reporter.success(f"distributed init_process", True)


def main_worker(rank, world_size, config):

    config = dotdict(config)

    config.rank = rank
    config.world_size = world_size

    # setup device id
    if config.ngpu > 0:
        config.device_id = config.rank
    else:
        config.device_id = -1

    reporter.success(f"rank {rank} world {world_size}")

    # initialize reporter
    reporter.init(config)

    reporter.success("after report")

    # create arch
    setup_process_group(config)

    # create pm, lm, am
    pm = read_pm(config.pm)
    config.pm = pm

    lm = read_lm(config.lm)
    config.lm = lm

    # prepare iterables
    tr_loader, cv_loader = read_loaders(config, pm, lm)

    if not config.debug_loader:

        # initialize task
        am = read_am(config.am)
        config.am = am
        am.config.rank = config.rank
        am.config.world_size = config.world_size

        trainer = Trainer(am, config)

        # start main logic
        reporter.info(f"start training...")

        trainer.train(tr_loader, cv_loader)

def main():

    config = create_am_config()
    config.mode = 'train'

    if config.model_path and str(config.model_path) != "none":
        verify_checkpoint_directory(Path(config.model_path))

    # deploy corpora to faster disk
    # if allosaurus_config.node_name.startswith('islpc'):
    #
    #     # only deploying with rank 0
    #     if config.rank == 0:
    #         for corpus in config.corpus_ids:
    #             deploy_corpus(corpus, config.feats, config.langs, force=config.force_deploy)

    world_size = config.ngpu

    reporter.info("start distributing")
    if world_size >= 2:
        mp.spawn(main_worker,
                 args=(world_size, dict(config)),
                 nprocs=world_size,
                 join=True)
    else:
        main_worker(0, world_size, config)


if __name__ == '__main__':
    main()