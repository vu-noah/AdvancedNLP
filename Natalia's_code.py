import stanza
import pandas as pd

nlp = stanza.Pipeline('en', download_method=None)

def extract_ner_and_phrase_type(filepath, outputpath):
    """
    Get NER labeles and store them in a .tsv file. 
    """
    
    df = pd.read_csv(filepath, sep='\t', header=None, names=['token_individual_id', 'token_global_id',
                                                             'token_id_in_sent', 'token', 'lemma',
                                                             'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                                                             'head_dependency_relation', 'additional_info',
                                                             'PB_predicate', 'semantic_role', 'is_candidate', 'sent_id',
                                                             'current_predicate', 'global_sent_id',
                                                             'candidate_prediction'],
                     quotechar='ą', engine='python')  
                                                             
    ner_lebels = [] 
    
    count = 0
            
    for group in df.groupby('sent_id', sort = False):
        sent_df = group[1]
        
        sentence = []
                
        for i, row in sent_df.iterrows():
                    
            sentence.append(row['token'])
                           
        count = count + 1
        print(f'sentence number: {count}')
        doc = nlp(' '.join(sentence))
        for i, token in enumerate(doc.sentences[0].tokens):
            ner_lebels.append(token.ner)
            
    ner_dict = {'ner_label':ner_lebels} 
    ner = pd.DataFrame(ner_dict) 
    ner.to_csv(outputpath, sep='\t', index=False)
    
if __name__ == '__main__':
    extract_ner_and_phrase_type('Data\train_data.tsv', 'Data\train_data_ner.tsv')
    extract_ner_and_phrase_type('Data\test_data.tsv', 'Data\test_data_ner.tsv')
