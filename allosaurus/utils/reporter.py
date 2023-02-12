from loguru import logger
import os
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import sys


class Reporter:

    def __init__(self):
        self.logger = logger
        self.writer = None
        self.rank = 0

        self.logger.remove()

        self.logger.add(sys.stdout, colorize=True, format="<green>{time:MM-DD HH:mm:ss}</green> <cyan>{level}</cyan> {message}", level="INFO")

        self.train_start_time = None
        self.validate_start_time = None


    def init(self, config):
        self.config = config
        self.rank = config.rank

        if "model_path" in config and config.model_path and config.model_path != "none":
            self.log_dir = Path(config.model_path) / 'log'
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.add(str(self.log_dir / 'log.txt'), encoding='utf-8')

        self.info(f"configure: \n {config}")

    def init_trainer(self, config):

        if config.rank == 0:
            if hasattr(config, "model_path") and config.model_path and config.model_path != "none":
                self.log_dir = Path(config.model_path) / 'log'
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self.log_dir)

        self.info(f"trainer from {config.rank}")

    def setup_distributed(self, config):
        self.config = config
        self.rank = config.rank

    def train(self):
        self.train_start_time = time.time()
        self.train_main_report = dict()
        self.train_total_iteration = 0

    def validate(self):
        self.validate_start_time = time.time()
        self.validate_main_report = dict()
        self.validate_total_iteration = 0


    def add_train_report(self, epoch, iteration, report):

        if self.rank == 0:
            cur_time = time.time()

            elapsed_time = cur_time - self.train_start_time

            cost_time = time.strftime("%H:%M", time.gmtime(elapsed_time))

            average_loss = report['average_loss']
            average_ter = report['average_ter']

            message = 'time {} epoch {} ({:08d}|{:08d}) speed: {:.2f} loss {:.4f} ter: {:.4f}'.format(
                cost_time, iteration, epoch, average_loss, average_ter)

            self.info(message)

            self.add_scalar('Train/Loss', average_loss, self.train_total_iteration)
            self.add_scalar('Train/TER', average_ter, self.train_total_iteration)

            self.train_iter += 1

    def add_validate_report(self, epoch, iteration, report):

        if self.rank == 0:
            cur_time = time.time()

            elapsed_time = cur_time - self.train_start_time

            cost_time = time.strftime("%H:%M", time.gmtime(elapsed_time))

            average_loss = report['average_loss']
            average_ter = report['average_ter']

            message = 'time {} epoch {} ({:08d}|{:08d}) speed: {:.2f} loss {:.4f} ter: {:.4f}'.format(
                cost_time, iteration, epoch, average_loss, average_ter)

            self.info(message)

            self.add_scalar('Validate/Loss', average_loss, self.validate_total_iteration)
            self.add_scalar('Validate/TER', average_ter, self.validate_total_iteration)

            self.train_iter += 1

    def add_figure(self, tag, figure):
        if self.rank == 0 and self.writer:
            self.writer.add_figure(tag, figure)

    def add_scalar(self, tag, scalar_value, global_step):
        if self.rank == 0 and self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def info(self, message, broadcast=False):
        if self.rank == 0 or broadcast:
            message = f'rank {self.rank}: {message}'
            self.logger.info(message)

    def debug(self, message, broadcast=False):
        if self.rank == 0 or broadcast:
            message = f'rank {self.rank}: {message}'
            self.logger.debug(message)

    def warning(self, message, broadcast=True):
        if self.rank == 0 or broadcast:
            message = f'rank {self.rank}: {message}'
            self.logger.warning(message)

    def success(self, message, broadcast=True):

        if self.rank == 0 or broadcast:
            message = f'rank {self.rank}: {message}'
            self.logger.success(message)
            if self.writer:
                self.writer.add_text('success', message)

    def critical(self, message, broadcast=True):

        if self.rank == 0 or broadcast:
            message = f'rank {self.rank}: {message}'
            self.logger.critical(message)
            if self.writer:
                self.writer.add_text('critical', message)

    def close(self):
        if self.writer:
            self.writer.close()



reporter = Reporter()