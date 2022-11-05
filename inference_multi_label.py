import pytorch_lightning as pl

from transformers import BertTokenizer
from models.multi_label_model import  MultiLabelModel

def inference(pretrained_model_dir, labels, threshold = 0.9):
    trained_model = MultiLabelModel.load_from_checkpoint(
        pretrained_model_dir,
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

    inference( pretrained_model_dir="./logs/bert_multilabel/version_1/checkpoints/epoch=5-step=792.ckpt",
               labels = labels )
