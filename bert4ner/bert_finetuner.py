### WARNING: Do NOT run this cell, unless you are running this on Google Colab. For a local installation run: pip
# install -r requirements.txt inside the terminal
# % pip install transformers==4.9.1
# % pip install datasets==1.11.0
# % pip install tabulate==0.8.9
# % pip install seqeval==1.2.2

import logging
import os
import random
import sys
import time
import bert_utils as utils
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import pipeline
from torch.utils.data import SequentialSampler

# Initialize Hyperparameters
EPOCHS = 2
BERT_MODEL_NAME = "bert-base-cased"
GPU_RUN_IX = 0

SEED_VAL = 1234500
SEQ_MAX_LEN = 256
PRINT_INFO_EVERY = 10  # Print status only every X batches
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4

TRAIN_DATA_PATH = "data/trial_mini_data.conll"  # "data/conll2003.train.conll"
DEV_DATA_PATH = "data/trial_mini_data.conll"  # "data/conll2003.dev.conll"

SAVE_MODEL_DIR = "saved_models/MY_BERT_NER/"

LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # -100

if not os.path.exists(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)

# Initialize Random seeds and validate if there's a GPU available...
device, USE_CUDA = utils.get_torch_device(GPU_RUN_IX)
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# Record everything inside a Log File
console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=f"{SAVE_MODEL_DIR}/BERT_TokenClassifier_train_{EPOCHS}.log")
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logging.info("Start Logging")

# Load Training and Validation  Datasets
# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

# Load Train Dataset
train_data, train_labels, train_label2index = utils.read_conll(TRAIN_DATA_PATH, has_labels=True)
train_inputs, train_masks, train_labels, seq_lengths = utils.data_to_tensors(train_data,
                                                                             tokenizer,
                                                                             max_len=SEQ_MAX_LEN,
                                                                             labels=train_labels,
                                                                             label2index=train_label2index,
                                                                             pad_token_label_id=PAD_TOKEN_LABEL_ID)
utils.save_label_dict(train_label2index, filename=LABELS_FILENAME)
index2label = {v: k for k, v in train_label2index.items()}

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Load Dev Dataset
dev_data, dev_labels, _ = utils.read_conll(DEV_DATA_PATH, has_labels=True)
dev_inputs, dev_masks, dev_labels, dev_lens = utils.data_to_tensors(dev_data,
                                                                    tokenizer,
                                                                    max_len=SEQ_MAX_LEN,
                                                                    labels=dev_labels,
                                                                    label2index=train_label2index,
                                                                    pad_token_label_id=PAD_TOKEN_LABEL_ID)

# Create the DataLoader for our Development set.
dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_lens)
dev_sampler = RandomSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

# Initialize Model Components
model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(train_label2index))
model.config.finetuning_task = 'token-classification'
model.config.id2label = index2label
model.config.label2id = train_label2index
if USE_CUDA:
    model.cuda()

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * EPOCHS

# Create optimizer and the learning rate scheduler.
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# Training Cycle(Fine - tunning)
loss_trn_values, loss_dev_values = [], []

for epoch_i in range(1, EPOCHS + 1):
    # Perform one full pass over the training set.
    logging.info("")
    logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
    logging.info('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        # Update parameters
        optimizer.step()
        scheduler.step()

        # Progress update
        if step % PRINT_INFO_EVERY == 0 and step != 0:
            # Calculate elapsed time in minutes.
            elapsed = utils.format_time(time.time() - t0)
            # Report progress.
            logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                            elapsed, loss.item()))

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_trn_values.append(avg_train_loss)

    logging.info("")
    logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
    logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

    # Validation
    # After the completion of each training epoch, measure our performance on our validation set.
    t0 = time.time()
    results, preds_list = utils.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label,
                                                    PAD_TOKEN_LABEL_ID, prefix="Validation Set")
    loss_dev_values.append(results['loss'])
    logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
    logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                                 results['recall'] * 100,
                                                                                 results['f1'] * 100))
    logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))

    # Save Checkpoint for this Epoch
    utils.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args": []}, model, tokenizer)

utils.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
utils.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
logging.info("")
logging.info("Training complete!")

# Use Fine - tuned Model for Predictions
GPU_IX = 0
device, USE_CUDA = utils.get_torch_device(GPU_IX)
FILE_HAS_GOLD = True
SEQ_MAX_LEN = 256
BATCH_SIZE = 4

TEST_DATA_PATH = "data/trial_mini_data.conll"  # "data/conll2003.dev.conll"
# TEST_DATA_PATH = "data/trial_unk_data.conll"
MODEL_DIR = "saved_models/MY_BERT_NER/"
LOAD_EPOCH = 1
INPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
OUTPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # -100

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/BERT_TokenClassifier_predictions.log")
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

# Load Pre - trained Model
model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
index2label = {v: k for k, v in label2index.items()}

# Load File for Predictions
test_data, test_labels, _ = utils.read_conll(TEST_DATA_PATH, has_labels=FILE_HAS_GOLD)
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

    logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label,
                                                    PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
    logging.info("  Test Loss: {0:.2f}".format(results['loss']))
    logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                                 results['recall'] * 100,
                                                                                 results['f1'] * 100))

    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for sent, pred in preds_list:
                fin.write(" ".join(sent) + "\n")
                fout.write(" ".join(pred) + "\n")

else:
    # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TokenClassificationPipeline
    logging.info('Predicting labels for {:,} test sentences...'.format(len(test_data)))
    if not USE_CUDA:
        GPU_IX = -1
    nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=GPU_IX)
    nlp.ignore_labels = []
    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for seq_ix, seq in enumerate(test_data):
                sentence = " ".join(seq)
                predicted_labels = []
                output_obj = nlp(sentence)
                # [print(o) for o in output_obj]
                for tok in output_obj:
                    if '##' not in tok['word']:
                        predicted_labels.append(tok['entity'])
                logging.info(f"\n----- {seq_ix + 1} -----\n{seq}\nPRED:{predicted_labels}")
                fin.write(sentence + "\n")
                fout.write(" ".join(predicted_labels) + "\n")
