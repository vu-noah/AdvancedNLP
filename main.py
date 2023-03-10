# 02.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train import fine_tune_bert
from predict import make_predictions_with_finetuned_model
import sys

def main(argv = None):
    if argv == None:
        argv = sys.argv
    epochs = int(argv[1])
    batch_size = int(argv[2])
    mode = argv[3]
    load_epoch = argv[4]
    has_gold = bool(argv[5])
    
    fine_tune_bert(epochs, batch_size, mode)
    make_predictions_with_finetuned_model(batch_size, load_epoch, has_gold, mode)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
   # command line example: 
   # python3 C:\\Users\\User\\AdvancedNLP\\main.py main 5 4 token_type_IDs 5 True
   
