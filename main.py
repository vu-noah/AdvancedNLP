# 02.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train import fine_tune_bert
from predict import make_predictions_with_finetuned_model

if __name__ == '__main__':
    fine_tune_bert(epochs=2, batch_size=4, mode='token_type_IDs')
    make_predictions_with_finetuned_model(batch_size=4, load_epoch=2, has_gold=True, mode='token_type_IDs')
