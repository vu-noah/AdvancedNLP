# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza
import re


def process_text_with_stanza(text):
    """
    Process text with an English stanza pipeline.
    :param str text: a text
    :return: stanza.Document processed_text
    """
    nlp = stanza.Pipeline('en')
    processed_text: stanza.Document = nlp(text)

    return processed_text


def extract_features(doc):
    """
    Extract feature dictionaries for every word in a stanza.Document object, store them in lists, zip and return them.
    :param stanza.Document doc: a stanza.Document object containing processed text
    :return: XXX
    """
    # create lists to store feature dictionaries in
    categorical_feature_dictionaries = []
    binary_feature_dictionaries = []

    for sentence in doc.sentences:
        # print(sentence)
        # constituent_tree = sentence.constituency
        # # print(help(constituent_tree))
        # # print(constituent_tree.label)
        # print(constituent_tree.children[0].children)
        # # print(constituent_tree.pretty_print())
        # # print(constituent_tree.leaf_labels())
        # # print(constituent_tree.depth())
        #
        # # constituents = str(constituent_tree.pretty_print()).split('\n')
        # # print(constituents)
        #
        # constituents = str(constituent_tree.children[0].children).replace(',', '')
        # constituents = constituents.replace('(', '', 1)
        # print(constituents)
        #
        # open_counter = 0
        # close_counter = 0
        # constituent = ''
        # for character in constituents:
        #     constituent = constituent + character
        #     if character == '(':
        #         open_counter += 1
        #     if character == ')':
        #         close_counter += 1
        #     if open_counter == close_counter:
        #         print(constituent)
        #         constituent = ''
        #         open_counter = 0
        #         close_counter = 0

        # constituents = []
        # def break_down_constituents(constituent_structure):
        #     for child in constituent_structure.children:
        #         constituent = str(child).split(')')
        #         constituents.append(constituent)
        #         break_down_constituents(child)
        #
        # break_down_constituents(constituent_tree.children[0])
        # for el in constituents:
        #     print(el)

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
    processed_text = process_text_with_stanza(text)
    feature_dictionaries = extract_features(processed_text)
    for categorical_binary_pair in feature_dictionaries:
        print(categorical_binary_pair)


if __name__ == '__main__':
    example_sentence = '''I love you.'''
    perform_feature_extraction(example_sentence)

# '''Everyone has the right to an effective remedy by the competent national tribunals for acts
#     violating the fundamental rights granted him by the constitution or by law.'''
