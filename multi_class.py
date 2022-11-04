from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    dm = PreprocessorClass(max_length = 50, 
                           preprocessed_dir = "data/multiclass/preprocessed",
                           batch_size = 100)
    
    model = MultiClassModel(
        n_out = 5,
        dropout = 0.3,
        lr = 1e-5
    )

    # 1e-3 = 0.0001
    # 2e-2 = 0.002

    logger = TensorBoardLogger("logs", name="bert-multi-class")

    trainer = pl.Trainer(
        gpus = 1,
        max_epochs = 10,
        default_root_dir = "./checkpoints/class",
        logger = logger
    )

    trainer.fit(model, datamodule = dm)
    pred, true = trainer.predict(model = model, datamodule = dm)
