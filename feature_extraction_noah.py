# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza
import re
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

    return processed_text_stanza


def obtain_information(tree_object):
    """

    :param tree_object:
    :return:
    """
    constituent_information = defaultdict(list)
    phrase_counter = defaultdict(int)

    def inspect_children(subtree, constituents, depth=-1):
        """

        :param subtree:
        :param constituents:
        :param depth:
        :return:
        """
        depth += 1
        for subtree in subtree.children:
            if subtree.label in re.findall(r'SBAR|NP|VP|PP|CC|ADJP|S|WHNP', subtree.label):
                phrase_counter[subtree.label] += 1
                phrase_counter_to_elements = {phrase_counter[subtree.label]: subtree.leaf_labels()}
                phrase_type_to_phrase_counter = {subtree.label: phrase_counter_to_elements}
                constituents[depth].append(phrase_type_to_phrase_counter)
                # phrase_number_level_information = f'{subtree.label}_num{phrase_counter[subtree.label]}_depth{depth}'
                # constituents[phrase_number_level_information] = subtree.leaf_labels()

            inspect_children(subtree, constituents, depth)

    # idea: nested dictionary instead having the information about depth and number of phrase in the key
    # {0: {S: {1: ['Marry', 'me', 'Juliet', ',', 'you', "'ll", 'never', 'have', 'to', 'be', 'alone', '.']}}}

    inspect_children(tree_object, constituent_information)
    print('RAW')
    print(constituent_information)
    print()
    print('PRETTY')
    for const, info in constituent_information.items():
        print(const, info)


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

        print(sentence.constituency.pretty_print())
        obtain_information(sentence.constituency)

        quit()

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
    processed_text = process_text(text)
    feature_dictionaries = extract_features(processed_text)
    for categorical_binary_pair in feature_dictionaries:
        print(categorical_binary_pair)


if __name__ == '__main__':
    example_sentence = '''I don\'t like you.'''
    perform_feature_extraction(example_sentence)

# '''Everyone has the right to an effective remedy by the competent national tribunals for acts
#     violating the fundamental rights granted him by the constitution or by law.'''
