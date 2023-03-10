# 02.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train import fine_tune_bert
from predict import make_predictions_with_finetuned_model
import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv
    epochs = int(argv[0])
    batch_size = int(argv[1])
    mode = argv[2]
    has_gold = bool(argv[3])
    
    fine_tune_bert(epochs=epochs, batch_size=batch_size, mode=mode)
    make_predictions_with_finetuned_model(batch_size=batch_size, load_epoch=epochs, has_gold=has_gold, mode=mode)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
   # command line example: 
   # python3 C:\\Users\\User\\AdvancedNLP\\main.py 5 4 token_type_IDs True
   
