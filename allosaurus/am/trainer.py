from allosaurus.utils.reporter import *
from allosaurus.utils.config import write_config
from allosaurus.utils.checkpoint_utils import *
from pathlib import Path
import torch.distributed as dist
from allosaurus.utils.meters import *
from torch.optim import Adadelta, Adam, SGD
from allosaurus.config import allosaurus_config
import time


class Trainer:

    def __init__(self, model, config):

        reporter.init_trainer(config)

        self.config = config

        # setup model_path
        if hasattr(self.config, "model_path") and self.config.model_path and self.config.model_path != "none":
            self.model_path = self.config.model_path

            if self.config.rank == 0:
                print(config)
                write_config(config, Path(self.model_path) / 'am.yml')

        else:
            self.model_path = "none"

        # number of no improvements
        self.num_no_improve = 0

        # accuracy for best arch
        self.best_ter = 3.14159265358979323846

        # move to cuda first
        if self.config.ngpu == 1:
            self.cuda = True
            self.model = model.cuda()
            self.model.device_id = 0
            reporter.success("moved arch and criterion to cuda")

        elif self.config.ngpu > 1:
            self.cuda = True
            self.model = model.distribute()
            reporter.success("distributed arch and criterion to cuda")

        else:
            self.cuda = False
            self.model = model

        # optimizer
        if "optimizer" in self.config:
            if self.config.optimizer == 'adadelta':
                reporter.success("using adadelta lr: "+str(self.config.lr_rate))
                self.optimizer = Adadelta(self.model.model.parameters(), lr=self.config.lr_rate, eps=1e-8,weight_decay=0.0)
            elif self.config.optimizer == 'adam':
                reporter.success("using adam lr: "+str(self.config.lr_rate))
                self.optimizer = Adam(self.model.model.parameters(), lr=self.config.lr_rate)
            elif self.config.optimizer == 'sgd':
                reporter.success("using sgd lr:"+str(self.config.lr_rate))
                self.optimizer = SGD(self.model.model.parameters(), lr=self.config.lr_rate)
        else:
            reporter.success("using default sgd lr:"+str(self.config.lr_rate))
            self.optimizer = SGD(self.model.model.parameters(), lr=self.config.lr_rate)

        # store the steps
        self.validate_iter = 0
        self.train_iter = 0

        # restore the arch
        if 'load_model'in config and config.load_model != 'none':
            reporter.info("Restoring " + str(self.config.load_model))
            torch_load(self.model.model, self.config.load_model)
        else:
            # load the pretrained ssl arch
            if 'ssl' in self.model.model.frontend.config and self.model.model.frontend.config.ssl == 'xlsr':
                reporter.info("loading xlsr_53_56k.pt")
                ckpt_state = torch.load(allosaurus_config.model_path / 'xlsr_53_56k.pt', map_location="cpu")
                self.model.model.frontend.model.load_state_dict(ckpt_state["model_weight"])

    def train(self, train_loader, cv_loader):

        # time stats
        start_time = time.time()
        prev_eval_time_stamp = start_time

        reporter.info(f"Total epoch {self.config.epoch}", True)
        reports = []

        if hasattr(self.config, "eval_first") and self.config.eval_first:
            reporter.info("evaluate first")
            self.validate(cv_loader)


        iter_time = 0
        train_time = 0

        for ii in range(self.config.epoch):

            train_iterator = iter(train_loader)
            iteration = len(train_iterator)

            # sample = next(train_iterator)

            # training steps
            for i in range(iteration):

                #cur_time = time.time()
                sample = next(train_iterator)
                #iter_time += time.time()-cur_time

                self.optimizer.zero_grad()

                #cur_time = time.time()
                report = self.model.train_step(sample)
                #train_time += time.time() - cur_time

                reports.append(report)

                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.grad_clip)

                self.optimizer.step()

                cur_time = time.time()

                # write ler to stdout for rank 0
                if i % self.config.report_per_batch == 0:

                    elapsed_time = cur_time - start_time
                    elapsed_time_from_last_epoch = cur_time - prev_eval_time_stamp

                    report, _ = self.model.reduce_report(reports)

                    total_token_size = report['total_token_size']
                    average_loss = report['average_loss']
                    average_ter = report['average_ter']
                    reports = []

                    cost_time = time.strftime("%H:%M", time.gmtime(elapsed_time))
                    message = 'time {} epoch {} ({:08d}|{:08d}) speed: {:.2f} loss {:.6f} ter: {:.6f}'.format(cost_time, ii, i,
                                iteration, total_token_size/elapsed_time_from_last_epoch, average_loss, average_ter)

                    reporter.info(message)
                    #print('train time: ', train_time / iteration, ' prep time: ', iter_time / iteration)
                    #print(list(self.arch.arch.named_parameters()))
                    #print(self.arch.arch.embed.weight)

                    reporter.add_scalar('Train/Loss', average_loss, self.train_iter)
                    reporter.add_scalar('Train/TER', average_ter, self.train_iter)

                    self.train_iter += 1

                # evaluation and dump after every constant period
                if (not self.config.eval_per_epoch) and (cur_time - prev_eval_time_stamp >= self.config.eval_per_second):

                    # update time
                    prev_eval_time_stamp = time.time()

                    if self.validate(cv_loader):
                        reporter.success("training finish")
                        return True

            # evaluation and dump after every constant period
            if self.config.eval_per_epoch:

                # update time
                prev_eval_time_stamp = time.time()

                if self.validate(cv_loader):
                    reporter.success("training finish")
                    return True

            #reporter.info(f"end epoch {ii}")

        reporter.success("all epoch done")

        return True

    def validate(self, cv_loader):

        reporter.info("------------------------------start validation--------------------------------------------------")

        iterator = iter(cv_loader)

        iteration = len(iterator)

        reports = []

        for i in range(iteration):

            sample = next(iterator)

            report = self.model.validate_step(sample)
            reports.append(report)

            if i % self.config.report_per_batch == 0:
                reporter.info('Validating ... ({:08d}|{:08d})'.format(i, iteration))

        total_report, corpus_report = self.model.reduce_report(reports)

        for corpus_id, report in corpus_report.items():

            average_ter  = report['average_ter']
            average_loss = report['average_loss']

            reporter.add_scalar("Validate/TER/"+corpus_id, average_ter, self.validate_iter)
            reporter.add_scalar("Validate/Loss/"+corpus_id, average_loss, self.validate_iter)

            message = "{:<20}  Validation TER: {}".format(corpus_id, average_ter, self.validate_iter)
            reporter.info(message)

        finish = False

        ter_ave = total_report['average_ter']

        message = "Average Validation TER: {}".format(ter_ave)
        reporter.success(message)

        reporter.add_scalar("Validate/TER", ter_ave, self.validate_iter)
        self.validate_iter += 1

        if ter_ave > self.best_ter:
            self.num_no_improve += 1

            # stop training if no improvement twice
            if self.num_no_improve >= 2 and self.config.nonstop == False:
                reporter.success("Stop training")
                finish = True

        # save arch on rank 0 if it is a better arch
        if ter_ave <= self.best_ter:
            # update best accuracy
            self.best_ler = ter_ave

            # restart no improve count
            self.num_no_improve = 0

            if self.model_path != 'none' and self.config.rank == 0:
                # save arch
                model_path = Path(self.model_path) / ("model_{:0.6f}.pt".format(ter_ave))

                torch_save(self.model.model, model_path)
                reporter.success(f"saved arch: {str(model_path)}")


        if self.config.world_size > 1:
            reporter.success(f"before barrier")
            dist.barrier()
            reporter.success(f"after barrier")

        # whether train should stop or not
        return finish

    def test(self, test_loader):

        iterator = iter(test_loader)

        iteration = len(iterator)

        reports = []

        for i in range(iteration):

            sample = next(iterator)

            report = self.model.test_step(sample)



        # whether train should stop or not
        return finish
