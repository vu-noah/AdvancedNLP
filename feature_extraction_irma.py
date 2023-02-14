# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza
import constituent_treelib
from constituent_treelib import ConstituentTree
from collections import defaultdict


def process_text(text):
    """
    Process text with an English stanza pipeline.
    :param str text: a text
    :return: stanza.Document processed_text_stanza, ConstituentTree constituent_tree
    """
    # instantiate stanza pipeline and process text
    stanza_pipeline = stanza.Pipeline('en')
    processed_text_stanza: stanza.Document = stanza_pipeline(text)

    # instantiate treelib pipeline (based on spaCy) and process text; afaik it works only for a single sentence, we
    # might need a loop for this one -> maybe we should tokenize our text with spaCy and then feed the pre-tokenized
    # text into stanza too (stanza can actually use the spaCy tokenizer) to make sure the tokenization is equal
    treelib_pipeline = ConstituentTree.create_pipeline(ConstituentTree.Language.English,
                                                       ConstituentTree.SpacyModelSize.Large)
    constituent_tree = ConstituentTree(text, treelib_pipeline)

    return processed_text_stanza, constituent_tree


def extract_features(doc, constituent_tree):
    """
    Extract feature dictionaries for every word in a stanza.Document object, store them in lists, zip and return them.
    :param stanza.Document doc: a stanza.Document object containing processed text
    :return: XXX
    """
    # create lists to store feature dictionaries in
    categorical_feature_dictionaries = []
    binary_feature_dictionaries = []

    for sentence in doc.sentences:
        for word in sentence.words:

            # create feature dictionaries for the word
            categorical_feature_dictionary = {'token': word.text.lower()}
            binary_feature_dictionary = {}

            # get the head of the token
            if word.head != 0:
                head_id = word.head-1
                categorical_feature_dictionary['head'] = sentence.words[head_id].text.lower()
            else:
                categorical_feature_dictionary['head'] = word.text.lower()

            # check whether the token is the root
            if word.head == 0:
                binary_feature_dictionary['is_root'] = 1
            else:
                binary_feature_dictionary['is_root'] = 0

            # get the constituencies ### exploratory, still in work
            print(constituent_tree)
            print(help(constituent_tree))
            constituents = constituent_tree.extract_all_phrases(
                content=constituent_treelib.core.ConstituentTree.NodeContent.Combined)
            print(constituents)

            new_dict = defaultdict(list)
            for label, c_list in constituents.items():
                for el in c_list:
                    new_dict[label].append(el.split())

            #print(new_dict)

            #quit()  # stop to test the code faster

            # append the feature dictionary to the list of feature dictionaries
            categorical_feature_dictionaries.append(categorical_feature_dictionary)
            binary_feature_dictionaries.append(binary_feature_dictionary)

    return zip(categorical_feature_dictionaries, binary_feature_dictionaries)


def perform_feature_extraction(text):
    """
    Extract feature dictionaries for every token from a text, print the feature dictionaries.
    :param text: a text
    :return: None
    """
    processed_text, constituent_tree = process_text(text)
    feature_dictionaries = extract_features(processed_text, constituent_tree)
    for categorical_binary_pair in feature_dictionaries:
        print(categorical_binary_pair)


if __name__ == '__main__':
    example_sentence = 'I love you from the bottom of my heart.'
    perform_feature_extraction(example_sentence)

# '''Everyone has the right to an effective remedy by the competent national tribunals for acts
#     violating the fundamental rights granted him by the constitution or by law.'''
