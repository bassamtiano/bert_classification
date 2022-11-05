import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.preprocessor_toxic import PreprocessorToxic
from models.multi_label_model import  MultiLabelModel

if __name__ == '__main__':

    dm = PreprocessorToxic()

    labels = [
        'HS', 
        'Abusive', 
        'HS_Individual', 
        'HS_Group', 
        'HS_Religion', 
        'HS_Race', 
        'HS_Physical', 
        'HS_Gender', 
        'HS_Other', 
        'HS_Weak', 
        'HS_Moderate', 
        'HS_Strong'
    ]

    model = MultiLabelModel(labels = labels)

    early_stop_callback = EarlyStopping(monitor = 'val_loss',
                                        min_delta = 0.00,
                                        patience = 3,
                                        mode = "min")

    logger = TensorBoardLogger("logs", name="bert_multilabel")

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 10,
                         logger = logger,
                         callbacks = [early_stop_callback])

    trainer.fit(model, datamodule = dm)
    trainer.predict(model = model, datamodule = dm)

    