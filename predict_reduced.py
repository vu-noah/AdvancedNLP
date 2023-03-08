# 02.03.2023
# Predict SR with a loaded model
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted

import bert_utils as utils
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import pipeline
from torch.utils.data import SequentialSampler


def make_predictions_with_finetuned_model():
    """

    :return:
    """
    # Use Fine - tuned Model for Predictions
    FILE_HAS_GOLD = True
    SEQ_MAX_LEN = 256
    BATCH_SIZE = 4

    TEST_DATA_PATH = "Data/mini_test.json"
    MODEL_DIR = "saved_models/MY_BERT_SRL/"
    LOAD_EPOCH = 10
    INPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # -100

    # Load Pre - trained Model
    model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v: k for k, v in label2index.items()}

    # Load File for Predictions
    test_data, test_labels, _ = utils.read_json_srl(TEST_DATA_PATH)
    prediction_inputs, prediction_masks, gold_labels, seq_lens = utils.data_to_tensors(test_data,
                                                                                       tokenizer,
                                                                                       max_len=SEQ_MAX_LEN,
                                                                                       labels=test_labels,
                                                                                       label2index=label2index)

    # Make Predictions
    if FILE_HAS_GOLD:
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, seq_lens)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer,
                                                        index2label, PAD_TOKEN_LABEL_ID, full_report=True,
                                                        prefix="Test Set")

        print("Test Loss: {0:.2f}".format(results['loss']))
        print("Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                            results['recall'] * 100,
                                                                            results['f1'] * 100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent) + "\n")
                    fout.write(" ".join(pred) + "\n")

    else:
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=-1)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, seq in enumerate(test_data):
                    sentence = " ".join(seq)
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    fin.write(sentence + "\n")
                    fout.write(" ".join(predicted_labels) + "\n")


if __name__ == '__main__':
    make_predictions_with_finetuned_model()
