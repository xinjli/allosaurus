from allosaurus.am.utils import move_to_tensor, torch_save
from allosaurus.am.criterion import read_criterion
from allosaurus.am.optimizer import read_optimizer
from allosaurus.am.reporter import Reporter
import editdistance
import numpy as np
import torch
from itertools import groupby
from allosaurus.model import get_model_path
import os
import json

class Trainer:

    def __init__(self, model, train_config):

        self.model = model
        self.train_config = train_config

        self.device_id = self.train_config.device_id

        # criterion, only ctc currently
        self.criterion = read_criterion(train_config)

        # optimizer, only sgd currently
        self.optimizer = read_optimizer(self.model, train_config)

        # reporter to write logs
        self.reporter = Reporter(train_config)

        # best per
        self.best_per = 100.0

        # intialize the model
        self.model_path = get_model_path(train_config.new_model)

        # counter for early stopping
        self.num_no_improvement = 0


    def sum_edit_distance(self, output_ndarray, output_lengths_ndarray, token_ndarray, token_lengths_ndarray):
        """
        compute SUM of ter in this batch

        """

        error_cnt_sum = 0.0

        for i in range(len(token_lengths_ndarray)):
            target_list = token_ndarray[i, :token_lengths_ndarray[i]].tolist()
            logit = output_ndarray[i][:output_lengths_ndarray[i]]

            raw_token = [x[0] for x in groupby(np.argmax(logit, axis=1))]
            decoded_token = list(filter(lambda a: a != 0, raw_token))

            error_cnt_sum += editdistance.distance(target_list, decoded_token)

        return error_cnt_sum


    def step(self, feat_batch, token_batch):

        # prepare torch tensors from numpy arrays
        feat_tensor, feat_lengths_tensor = move_to_tensor(feat_batch, self.device_id)
        token_tensor, token_lengths_tensor = move_to_tensor(token_batch, self.device_id)

        #print(feat_tensor)
        #print(feat_lengths_tensor)
        output_tensor = self.model(feat_tensor, feat_lengths_tensor)

        #print(output_tensor)
        #print(token_tensor)
        #print(token_lengths_tensor)

        loss = self.criterion(output_tensor, feat_lengths_tensor, token_tensor, token_lengths_tensor)
        #print(loss.item())

        # extract numpy format for edit distance computing
        output_ndarray = output_tensor.cpu().detach().numpy()
        feat_ndarray, feat_lengths_ndarray = feat_batch
        token_ndarray, token_lengths_ndarray = token_batch

        phone_error_sum = self.sum_edit_distance(output_ndarray, feat_lengths_ndarray, token_ndarray,
                                                 token_lengths_ndarray)

        phone_count = sum(token_lengths_ndarray)

        return loss, phone_error_sum, phone_count


    def train(self, train_loader, validate_loader):

        self.best_per = 100.0

        batch_count = len(train_loader)

        for epoch in range(self.train_config.epoch):

            # shuffle
            train_loader.shuffle()

            # set to the training mode
            self.model.train()

            # reset all stats
            all_phone_count = 0.0
            all_loss_sum = 0.0
            all_phone_error_sum = 0.0

            # training loop
            for ii in range(batch_count):

                self.optimizer.zero_grad()

                feat_batch, token_batch = train_loader.read_batch(ii)

                # forward step
                loss_tensor, phone_error_sum, phone_count = self.step(feat_batch, token_batch)

                # backprop and optimize
                loss_tensor.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)

                self.optimizer.step()

                # update stats
                loss_sum = loss_tensor.item()
                all_phone_count += phone_count
                all_loss_sum += loss_sum
                all_phone_error_sum += phone_error_sum

                if ii % self.train_config.report_per_batch == 0:
                    message = f'epoch[batch]: {epoch:02d}[{ii:04d}] | train loss {all_loss_sum/all_phone_count:0.5f} train per {all_phone_error_sum / all_phone_count:0.5f}'
                    self.reporter.write(message)

                    # reset all stats
                    all_phone_count = 0.0
                    all_loss_sum = 0.0
                    all_phone_error_sum = 0.0


            # evaluate this model
            validate_phone_error_rate = self.validate(validate_loader)

            self.reporter.write(f"epoch{epoch} | validate per : {validate_phone_error_rate:0.5f}")
            if validate_phone_error_rate <= self.best_per:
                self.best_per = validate_phone_error_rate
                self.num_no_improvement = 0
                self.reporter.write("saving model")

                model_name = f"model_{validate_phone_error_rate:0.5f}.pt"

                # save model
                torch_save(self.model, self.model_path / model_name)

                # overwrite the best model
                torch_save(self.model, self.model_path / 'model.pt')

            else:
                self.num_no_improvement += 1

                if self.num_no_improvement >= 3:
                    self.reporter.write("no improvements for several epochs, early stopping now")
                    break

        # close reporter stream
        self.reporter.close()


    def validate(self, validate_loader):

        self.model.eval()

        batch_count = len(validate_loader)

        all_phone_error_sum = 0
        all_phone_count = 0

        # validation loop
        for ii in range(batch_count):

            self.optimizer.zero_grad()

            feat_batch, token_batch = validate_loader.read_batch(ii)

            # one step
            loss_tensor, phone_error_sum, phone_count = self.step(feat_batch, token_batch)

            # update stats
            all_phone_error_sum += phone_error_sum
            all_phone_count += phone_count

        return all_phone_error_sum/all_phone_count
