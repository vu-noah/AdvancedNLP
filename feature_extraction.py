# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza


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
    Extract feature dictionaries for every word in a stanza.Document object, store them in a list, return them.
    :param stanza.Document doc: a stanza.Document object containing processed text
    :return: list[dict[str, Any]] feature_dictionaries
    """
    # create a list to store feature dictionaries in
    feature_dictionaries = []

    for sentence in doc.sentences:
        for word in sentence.words:

            # create a feature dictionary for the word
            feature_dictionary = {'token': word.text.lower()}

            # get the head of the token
            head_id = word.head-1
            feature_dictionary['head'] = sentence.words[head_id].text.lower()

            # append the feature dictionary to the list of feature dictionaries
            feature_dictionaries.append(feature_dictionary)

    return feature_dictionaries


def main(text):
    """
    Extract feature dictionaries for every token from a text, print the feature dictionaries.
    :param text: a text
    :return: None
    """
    processed_text = process_text_with_stanza(text)
    feature_dictionaries = extract_features(processed_text)
    print(feature_dictionaries)


if __name__ == '__main__':
    example_sentence = '''Everyone has the right to an effective remedy by the competent national tribunals for acts 
    violating the fundamental rights granted him by the constitution or by law.'''
    main(example_sentence)