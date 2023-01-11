import glob
from os.path import join, getctime
import torch

class ModelSaver(object):
    def __init__(self,
                 log_dir,
                 device,
                 model) -> None:
        self.log_dir = log_dir
        self.device = device
        self.model = model

    def save_trained_model(self, trained_model, epoch = None):
        epoch = "_epoch_{}".format(epoch) if epoch is not None else ""
        p = join(self.log_dir, 'trained_model{}.pt'.format(epoch))
        torch.save(trained_model.state_dict(), p)
        print('Trained model saved to {}'.format(p))

    def load_trained_model(self, trained_model_dir):
        p = join(self.log_dir, 'trained_model*')
        files = glob.glob(p)
        best_trained_model_path = max(files, key=getctime)
        trained_model = self.model
        trained_model.load_state_dict(
            torch.load(best_trained_model_path, map_location=self.device))
        trained_model.to(self.device)
        return trained_model