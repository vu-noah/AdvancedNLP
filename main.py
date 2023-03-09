# 02.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train import fine_tune_bert
from predict import make_predictions_with_finetuned_model

if __name__ == '__main__':
    fine_tune_bert(epochs=5, batch_size=4)
    make_predictions_with_finetuned_model()
