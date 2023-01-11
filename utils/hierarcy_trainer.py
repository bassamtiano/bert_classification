import sys
from statistics import mean

import torch
import torch.nn as nn

from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from models.hierarchy import Hierarchy

class HierarcyTrainer(object):
    def __init__(self,
                 level,
                 preprocess,
                 embedding_dim,
                 in_channels,
                 out_channels,
                 num_classes,
                 dropout,
                 lr,
                 device,
                 max_epoch,
                 last_level_checkpoint = None) -> None:
        super(HierarcyTrainer, self).__init__()

        torch.manual_seed(25)
        torch.cuda.manual_seed(25)
        torch.cuda.manual_seed_all(25)
        torch.backends.cudnn.deterministic = True

        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch

        self.train_data, self.val_data = preprocess.preprocessor(level = level, train_status = "train")

        self.model = Hierarchy(lr,
                               embedding_dim,
                               in_channels,
                               out_channels,
                               num_classes,
                               dropout)

        self.model.to(self.device)

        if last_level_checkpoint:
            self.model.load_state_dict(torch.load(last_level_checkpoint))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                           start_factor = 0.5, 
                                                           total_iters = 100)
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def scoring_helper(self, true, pred):
        report = classification_report(true, pred, output_dict = True, zero_division = 0)["accuracy"]
        f1_micro = round(f1_score(true, pred, average="micro"), 2)
        f1_macro = round(f1_score(true, pred, average="macro"), 2)

        accuracy = round(report, 2)

        return accuracy, f1_micro, f1_macro



    def train_step(self):
        self.model.train()
        self.model.zero_grad()

        step_loss = []
        step_accuracy = []

        progress_train = tqdm(self.train_data)

        for batch in progress_train:
            x_input_ids, y = batch

            x_input_ids = x_input_ids.to(self.device)
            y = y.to(self.device)

            out = self.model(input_ids = x_input_ids)

            loss = self.criterion(out, target = y.float())

            step_loss.append(loss.item())

            pred = out.argmax(1).cpu()
            true = y.argmax(1).cpu()

            accuracy, f1_micro, f1_macro = self.scoring_helper(true, pred)



            step_accuracy.append(accuracy)

            progress_train.set_description("loss : " + str(round(loss.item(), 2)) + 
                                           " | acc : " + str(accuracy) + 
                                           " | f1 micro : " + str(f1_micro) +
                                           " | f1 macro : " + str(f1_macro))

            loss.backward()
            self.optimizer.step()

        print("Train Step Loss : ", round(mean(step_loss), 2))
        print("Train Step Accuracy : ", round(mean(step_accuracy), 2))

        self.scheduler.step()
        return mean(step_loss)

    def validation_step(self):
        pass

    def training(self):
        for i in range(self.max_epoch):
            print("Step ", (i+1) )
            print("="*50)
            print("Training Step")
            loss = self.train_step()


            print("="*50)