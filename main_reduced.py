# 07.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train_reduced import fine_tune_bert
from predict_reduced import make_predictions_with_finetuned_model

if __name__ == '__main__':
    fine_tune_bert(epochs=1, batch_size=4, mode='token_type_IDs')
    make_predictions_with_finetuned_model(mode='token_type_IDs')
