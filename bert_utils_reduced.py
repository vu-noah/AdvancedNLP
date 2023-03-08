# 02.03.2023
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted

import json
import os
import re
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.utils.dummy_pt_objects import BertModel

LongTensor = torch.LongTensor
device = torch.device("cpu")


##### Data Loading Functions #####
def wordpieces_to_tokens(wordpieces: list, labelpieces: list = None) -> tuple[list, list]:
    """

    :param wordpieces:
    :param labelpieces:
    :return:
    """
    textpieces = " ".join(wordpieces)
    full_words = re.sub(r'\s##', '', textpieces).split()
    full_labels = []
    if labelpieces:
        for ix, wp in enumerate(wordpieces):
            if not wp.startswith('##'):
                full_labels.append(labelpieces[ix])
        assert len(full_words) == len(full_labels)

    return full_words, full_labels


def expand_to_wordpieces(original_sentence: list, tokenizer: BertTokenizer, original_labels: list = None) \
        -> tuple[list, list]:
    """
    Also Expands BIO, but assigns the original label ONLY to the Head of the WordPiece (First WP)
    :param original_sentence: List of Full-Words
    :param original_labels: List of Labels corresponding to each Full-Word
    :param tokenizer: To convert it into BERT-model WordPieces
    :return:
    """
    txt_sentence = " ".join(original_sentence)
    txt_sentence = txt_sentence.replace("##", "")
    word_pieces = tokenizer.tokenize(txt_sentence)

    if original_labels:
        tmp_labels, lbl_ix = [], 0
        head_tokens = [1] * len(word_pieces)
        for i, tok in enumerate(word_pieces):
            if "##" in tok:
                tmp_labels.append("X")
                head_tokens[i] = 0
            else:
                tmp_labels.append(original_labels[lbl_ix])
                lbl_ix += 1

        word_pieces = ["[CLS]"] + word_pieces + ["[SEP]"]
        labels = ["X"] + tmp_labels + ["X"]

        return word_pieces, labels

    else:
        return word_pieces, []


def data_to_tensors(dataset: list, tokenizer: BertTokenizer, max_len: int, labels: list = None,
                    label2index: dict = None, pad_token_label_id: int = -100) -> tuple:
    """

    :param dataset:
    :param tokenizer:
    :param max_len:
    :param labels:
    :param label2index:
    :param pad_token_label_id:
    :return:
    """
    tokenized_sentences, label_indices = [], []

    for i, sentence in enumerate(dataset):
        # Get WordPiece Indices
        if labels and label2index:
            wordpieces, labelset = expand_to_wordpieces(sentence, tokenizer, labels[i])
            label_indices.append([label2index.get(lbl, pad_token_label_id) for lbl in labelset])
        else:
            wordpieces, labelset = expand_to_wordpieces(sentence, tokenizer, None)
        input_ids = tokenizer.convert_tokens_to_ids(wordpieces)
        tokenized_sentences.append(input_ids)

    seq_lengths = [len(s) for s in tokenized_sentences]

    # PAD ALL SEQUENCES
    input_ids = pad_sequences(tokenized_sentences, maxlen=max_len, dtype="long", value=0, truncating="post",
                              padding="post")
    if label_indices:
        label_ids = pad_sequences(label_indices, maxlen=max_len, dtype="long", value=pad_token_label_id,
                                  truncating="post", padding="post")
        label_ids = LongTensor(label_ids)
    else:
        label_ids = None
    # Create attention masks
    attention_masks = []
    # For each sentence...
    for i, sent in enumerate(input_ids):
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return LongTensor(input_ids), LongTensor(attention_masks), label_ids, LongTensor(seq_lengths)


def get_annotatated_sentence(rows: list, has_labels: bool) -> tuple[list, list]:
    """

    :param rows:
    :param has_labels:
    :return:
    """
    x, y = [], []

    for row in rows:
        if has_labels:
            tok, chunk, chunk_bio, ent_bio = row
            x.append(tok)
            y.append(ent_bio)
        else:
            tok, chunk, chunk_bio, _ = row
            x.append(tok)

    return x, y


def add_to_label_dict(labels: list, label_dict: dict) -> dict:
    """

    :param labels:
    :param label_dict:
    :return:
    """
    for l in labels:
        if l not in label_dict:
            label_dict[l] = len(label_dict)

    return label_dict


def read_json_srl(filename: str) -> tuple[list[list], list[list], dict]:
    """
    Read in a json file created from an original conllu file and extract the tokens and labels for each sentence as well
    as a dictionary mapping the labels to a number. Flag the current predicate.

    :param str filename: the path to the json file you want to read in
    :return: all_sentences, all_labels, label_dict
    """
    all_sentences, all_labels, label_dict = [], [], {}

    with open(filename) as infile:
        for line in infile.readlines():
            line = line.strip('\n')
            sentence_information = json.loads(line)

            sentence = sentence_information['seq_words']
            labels = sentence_information['BIO']
            predicate_index = sentence_information['pred_sense'][0]

            flagged_sentence = sentence[:predicate_index] + ['[PRED]'] + [sentence[predicate_index]] + ['[\PRED]'] + \
                               sentence[predicate_index + 1:]

            flagged_labels = labels[:predicate_index] + ['O'] + [labels[predicate_index]] + ['O'] + \
                             labels[predicate_index + 1:]

            all_sentences.append(flagged_sentence)
            all_labels.append(flagged_labels)

            label_dict = add_to_label_dict(flagged_labels, label_dict)

    return all_sentences, all_labels, label_dict


##### Evaluation Functions #####
def evaluate_bert_model(eval_dataloader: DataLoader, eval_batch_size: int, model: BertModel, tokenizer: BertTokenizer,
                        label_map: dict, pad_token_label_id: int, full_report: bool = False, prefix: str = "") \
        -> tuple[dict, list]:
    """

    :param eval_dataloader:
    :param eval_batch_size:
    :param model:
    :param tokenizer:
    :param label_map:
    :param pad_token_label_id:
    :param full_report:
    :param prefix:
    :return:
    """
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    input_ids, gold_label_ids = None, None
    # Put model on Evaluation Mode!
    model.eval()
    for batch in eval_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_len = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            gold_label_ids = b_labels.detach().cpu().numpy()
            input_ids = b_input_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            gold_label_ids = np.append(gold_label_ids, b_labels.detach().cpu().numpy(), axis=0)
            input_ids = np.append(input_ids, b_input_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    gold_label_list = [[] for _ in range(gold_label_ids.shape[0])]
    pred_label_list = [[] for _ in range(gold_label_ids.shape[0])]
    full_word_preds = []

    for seq_ix in range(gold_label_ids.shape[0]):
        for j in range(gold_label_ids.shape[1]):
            if gold_label_ids[seq_ix, j] != pad_token_label_id:
                gold_label_list[seq_ix].append(label_map[gold_label_ids[seq_ix][j]])
                pred_label_list[seq_ix].append(label_map[preds[seq_ix][j]])

        if full_report:
            wordpieces = tokenizer.convert_ids_to_tokens(input_ids[seq_ix], skip_special_tokens=True)
            full_words, _ = wordpieces_to_tokens(wordpieces, labelpieces=None)
            full_preds = pred_label_list[seq_ix]
            full_gold = gold_label_list[seq_ix]
            full_word_preds.append((full_words, full_preds))

    results = {
        "loss": eval_loss,
        "precision": precision_score(gold_label_list, pred_label_list),
        "recall": recall_score(gold_label_list, pred_label_list),
        "f1": f1_score(gold_label_list, pred_label_list),
    }

    if full_report:
        print("\n\n" + classification_report(gold_label_list, pred_label_list))

    return results, full_word_preds


##### Input/Output Functions #####
def save_losses(losses: dict, filename: str) -> None:
    """

    :param losses:
    :param filename:
    :return:
    """
    out = open(filename, "w")
    out.write(json.dumps({"losses": losses}) + "\n")


def save_label_dict(label2index: dict, filename: str) -> None:
    """

    :param label2index:
    :param filename:
    :return:
    """
    out = open(filename, "w")
    out.write(json.dumps(label2index))


def load_label_dict(modelpath: str) -> dict:
    """

    :param modelpath:
    :return:
    """
    fp = open(modelpath)
    label_dict = json.load(fp)

    return label_dict


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
def save_model(output_dir: str, arg_dict: dict, model: BertModel, tokenizer: BertTokenizer):
    """

    :param output_dir:
    :param arg_dict:
    :param model:
    :param tokenizer:
    :return:
    """
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(arg_dict, os.path.join(output_dir, 'training_args.bin'))


def load_model(model_class, tokenizer_class, model_dir) -> tuple:
    """

    :param model_class:
    :param tokenizer_class:
    :param model_dir:
    :return:
    """
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    # Copy the model to the GPU.
    model.to(device)

    return model, tokenizer


##### Misc Functions #####
def get_bool_value(str_bool: str) -> bool:
    """

    :param str_bool:
    :return:
    """
    if str_bool.upper() == "TRUE" or str_bool.upper() == "T":
        return True
    else:
        return False
