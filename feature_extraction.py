# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza
import re
from collections import defaultdict
from lxml import etree


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


def parse_string_to_xml(sentence_list, node, constituent_tree_object):
    """

    :param sentence_list:
    :param node:
    :param constituent_tree_object:
    :return:
    """
    for child in constituent_tree_object.children:
        if len(str(child).split(' ')) == 2:
            element = etree.SubElement(node, 'terminal')
            element.set('POS', child.label)
            word = child.leaf_labels()[0]
            index_of_word_in_sentence = sentence_list.index(word)
            element.set('INDEX', str(index_of_word_in_sentence))
            sentence_list[index_of_word_in_sentence] = 'ALR_PARSED'
            element.text = child.leaf_labels()[0]
            continue
        else:
            try:
                element = etree.SubElement(node, child.label)
            except ValueError:
                element = etree.SubElement(node, 'PUNCT')
            if len(str(child).split(' ')) == 2:
                element.text = child.leaf_labels()[0]
                continue

        parse_string_to_xml(sentence_list, element, child)

    return node


def get_phrase_type(tree, word):
    """

    :param tree:
    :param word:
    :return:
    """
    for element in tree.iter():
        if element.text == word.text and int(element.get('INDEX')) == word.id-1:
            parent = (element.getparent())

            return parent.tag


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

        sentence_token_list = [word.text for word in sentence.words]
        root = etree.Element("sentence")
        tree = parse_string_to_xml(sentence_token_list, root, sentence.constituency)
        etree.dump(tree)

        for word in sentence.words:

            # create feature dictionaries for the word
            categorical_feature_dictionary = {'word': word.text.lower()}
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

            # get the phrase type of the phrase the current token belongs to
            categorical_feature_dictionary['phrase_type'] = get_phrase_type(tree, word)

            # get whole constituent (words and POS)
            constituent_tokens = []
            constituent_pos = []
            for element in tree.iter():
                if element.text == word.text and int(element.get('INDEX')) == word.id-1:
                    parent = element.getparent()
                    for element_2 in parent.findall("terminal"):
                        constituent_tokens.append(element_2.text.lower())
                        constituent_pos.append(element_2.attrib['POS'])

            categorical_feature_dictionary['constituent_words'] = constituent_tokens
            categorical_feature_dictionary['constituent_POS'] = constituent_pos

            # if NP, governed by what 'S' or 'VP'?
            for element in root.findall('.//terminal'):
                if element.getparent().tag == 'NP' and element.text == word.text:
                    if element.getparent() not in root.find('.//VP').findall('.//NP'):
                        categorical_feature_dictionary['governed_by'] = 'S'
                    else:
                        categorical_feature_dictionary['governed_by'] = 'VP'

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
    example_sentence = '''The chubby llama is eating a bunch of grass.'''
    perform_feature_extraction(example_sentence)

# '''Everyone has the right to an effective remedy by the competent national tribunals for acts
#     violating the fundamental rights granted him by the constitution or by law.'''
