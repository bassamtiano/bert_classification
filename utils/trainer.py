import sys
from statistics import mean
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report
from tqdm import tqdm

from models.multi_class_only import MultiClassOnly
from utils.model_saver import ModelSaver

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
        

        self.model_saver = ModelSaver(log_dir = "logs/manual_bert", device = device, model = self.model)
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
        step_loss = []
        step_accuracy = []

        progress_train = tqdm(self.train_data)

        for batch in progress_train:
            x_input_ids, x_token_type_ids, x_attention_mask, y = batch


            x_input_ids = x_input_ids.to(self.device)           
            y = y.to(self.device)


            out = self.model(input_ids = x_input_ids, 
                             attention_mask = x_attention_mask,
                             token_type_ids = x_token_type_ids)
            loss = self.criterion(out, target = y.float())
            # print(loss.item())
            step_loss.append(loss.item())

            pred = out.argmax(1).cpu()
            true = y.argmax(1).cpu()
            report = classification_report(true, pred, output_dict = True, zero_division = 0)
            step_accuracy.append(report["accuracy"])

            progress_train.set_description("loss : " + str(round(loss.item(), 2)) + " | acc : " + str(round(report["accuracy"], 2)))

            loss.backward()
            self.optimizer.step()

        print("Train Step Loss : ", round(mean(step_loss), 2))
        print("Train Step Accuracy : ", round(mean(step_accuracy), 2))

        self.scheduler.step()
        return mean(step_loss)
    
    def validation_step(self):
        with torch.no_grad():
            step_loss = []
            step_accuracy = []

            progress_val = tqdm(self.val_data)

            for batch in progress_val:
                x_input_ids, x_token_type_ids, x_attention_mask, y = batch


                x_input_ids = x_input_ids.to(self.device)           
                y = y.to(self.device)


                out = self.model(input_ids = x_input_ids, 
                                attention_mask = x_attention_mask,
                                token_type_ids = x_token_type_ids)
                loss = self.criterion(out, target = y.float())
                # print(loss.item())
                step_loss.append(loss.item())

                pred = out.argmax(1).cpu()
                true = y.argmax(1).cpu()
                report = classification_report(true, pred, output_dict = True, zero_division = 0)
                step_accuracy.append(report["accuracy"])

                progress_val.set_description("loss : " + str(round(loss.item(), 2)) + " | acc : " + str(round(report["accuracy"], 2)))

            print("Val Step Loss : ", round(mean(step_loss), 2))
            print("Val Step Accuracy : ", round(mean(step_accuracy), 2))

            self.scheduler.step()
            return mean(step_loss)

    def training(self):
        for i in range(100):
            print("Step ", (i+1) )
            print("="*50)
            print("Training Step")
            loss = self.train_step()

            print("Validation Step")
            loss = self.validation_step()

            print("Save model")
            self.model_saver.save_trained_model(trained_model = self.model, 
                                                epoch = i + 1)

            print("="*50)
