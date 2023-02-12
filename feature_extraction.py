# 12.02.2023
# Advanced NLP - Assignment 1
# Feature Extraction

import stanza

# example sentence we can work with
example_sentence = '''Everyone has the right to an effective remedy by the competent national tribunals for acts 
violating the fundamental rights granted him by the constitution or by law.'''

# prepare the pipeline and process the example sentence
nlp = stanza.Pipeline('en')
doc = nlp(example_sentence)

def extract_basic_features():
    for sentence in doc.sentences:
    # print(sentence)
    for word in sentence.words:
        # create a feature dictionary for the word
        feature_dictionary = {''}
        head_id = word.head - 1
        print(word.text, 'is', word.deprel, 'to', sentence.words[head_id].text)
