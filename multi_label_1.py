import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer

from utils.preprocessor_toxic import PreprocessorToxic
from models.multi_label_model import MultiLabelModel


def predictor(trainer, labels, threshold = 0.5):
    trained_model = MultiLabelModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        labels = labels
    )
    trained_model.eval()
    trained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    test_comment = "sih kerja delay mulu setan"

    encoding = tokenizer.encode_plus(
        test_comment,
        add_special_tokens = True,
        max_length = 100,
        return_token_type_ids = True,
        padding = "max_length",
        return_attention_mask = True,
        return_tensors = 'pt',
    )

    test_prediction = trained_model(encoding["input_ids"], 
                                    encoding["token_type_ids"],
                                    encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()

    for label, prediction in zip(labels, test_prediction):
        if prediction < threshold:
            continue
        print(f"{label}: {prediction}")

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

    model = MultiLabelModel(labels)

    early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                        min_delta = 0.00,
                                        patience = 3,
                                        mode = "min")
    
    logger = TensorBoardLogger("logs", name="bert_multilabel")

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 10,
                         logger = logger,
                         default_root_dir = "./checkpoints/labels",
                         callbacks = [early_stop_callback])

    trainer.fit(model, datamodule = dm)
    trainer.predict(model = model, datamodule = dm)

    print("Predictor")
    predictor(trainer, labels)