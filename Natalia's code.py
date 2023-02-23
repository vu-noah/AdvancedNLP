import stanza
import pandas as pd

nlp = stanza.Pipeline('en', download_method=None)

filepath = r'Data\test_data.tsv'

df = pd.read_csv(filepath, sep='\t', header=None, names=['token_global_id', 'token_id_in_sent', 'token', 'lemma',
                                                         'upos', 'pos', 'grammar', 'head_id', 'dependency_label',
                                                         'head_dependency_relation', 'additional_info',
                                                         'proposition', 'semantic_role', 'is_candidate', 'sent_id',
                                                         'candidate_prediction'])

def get_ner_type(df):
    """
    Get NER type for each token. 
    
    :param df: pandas DataFrame 
    :return: a list of labels 
    """

    tokens = list(df['token'])
    print('started compiling')
    doc = nlp(' '.join(tokens))
    print('finished compiling')
    ner_lebels = []
    for i, token in enumerate(doc.sentences[0].tokens):
        ner_lebels.append(token.ner)
        
    return ner_lebels
        
    print(len(tokens)) 
    print(len(ner_lebels)) 
    
    ner_dict = {'token':tokens,'ner_label':ner_lebels} 
    ner = pd.DataFrame(ner_dict) 
    ner.to_csv('stanza_ner_labels_test.tsv', sep='\t', index=False) 
    
    
def get_ner_type_2(df):
    """
    Do the same but for each setence instead of the whole document 
    """
    tokens = list(df['token'])
    tokens_ids = list(df['token_id_in_sent'])
    correct_token_ids = []
    for id in tokens_ids:
        correct_token_ids.append(id - 1)
        
    current_sentence, sentences = [], []

    for i, token in enumerate(tokens):
        current_sentence.append(token)
        if i == len(tokens) - 1 or correct_token_ids[i+1] == 0:
            sentences.append(current_sentence)
            current_sentence = []

    ner_lebels = []        
    for i, sentence in enumerate(sentences):
        print(f'sentence number: {i}')
        doc = nlp(' '.join(sentence))
        for i, token in enumerate(doc.sentences[0].tokens):
            ner_lebels.append(token.ner)
        
    return ner_lebels

    print(len(tokens)) 
    print(len(ner_lebels)) 
    
    ner_dict = {'token':tokens,'ner_label':ner_lebels} 
    ner = pd.DataFrame(ner_dict) 
    ner.to_csv('stanza_ner_labels_test_2.tsv', sep='\t', index=False) 