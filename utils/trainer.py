import sys

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_class_only import MultiClassOnly


class Trainer(object):
    def __init__(self, 
                 preprocessor,
                 device,
                 n_out,
                 dropout,
                 lr) -> None:
        super(Trainer, self).__init__()

        torch.manual_seed(25)
        torch.cuda.manual_seed(25)
        torch.cuda.manual_seed_all(25)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

        self.device = device
        self.lr = lr

        self.train_data, self.val_data, self.test_data = preprocessor.preprocessor_manual()

        self.model = MultiClassOnly(n_out = n_out, dropout = dropout)
        self.model.to(self.device)
        
        train_total = len(self.train_data) * 10

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                           start_factor = 0.5, 
                                                           total_iters = 100)
        


        self.criterion = nn.BCEWithLogitsLoss()


    def get_constant_schedule_with_warmup(self, optimizer, num_warmup_steps, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def train_step(self):
        self.model.train()
        self.model.zero_grad()

        for i, batch in enumerate(self.train_data):
            x_input_ids, x_token_type_ids, x_attention_mask, y = batch


            x_input_ids = x_input_ids.to(self.device)           
            y = y.to(self.device)


            out = self.model(input_ids = x_input_ids, 
                             attention_mask = x_attention_mask,
                             token_type_ids = x_token_type_ids)
            loss = self.criterion(out, target = y.float())
            print(loss.item())
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        return loss.item()
            


    def training(self):
        for i in range(100):
            loss = self.train_step()
            print("===")
            # print(loss)
