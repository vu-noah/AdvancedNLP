from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack


def concatenate_categorical_and_numerical_feature_vectors(categorical_vectors_matrix, numerical_vectors_list):
    """
    Concatenate the vectors obtained from DictVectorizer and the list of numerical feature vectors.
    :param csr_matrix categorical_vectors_matrix: a matrix holding feature vectors compiled with DictVectorizer
    :param list numerical_vectors_list: a list holding vectors for numerical features
    :return: csr_matrix full_feature_vecs: a matrix of concatenated feature vectors
    """
    full_feature_vecs = hstack([categorical_vectors_matrix, numerical_vectors_list], format='csr')

    return full_feature_vecs


def run_logreg(X_train_cat_dicts, X_train_num_dicts, y_train, X_test_cat_dicts, X_test_num_dicts, y_test, df_test,
               step):
    """
    Run Logistic Regression model.   
    
    :param list[dict] X_train_cat_dicts: a list (of dicts) containing categorical features from training data
    :param list[dict] X_train_num_dicts: a list (of dicts) containing numerical features from training data
    :param list y_train: a list of gold labels from training data
    :param list[dict] X_test_cat_dicts: a list (of dicts) containing categorical features from test data
    :param list[dict] X_test_num_dicts: a list (of dicts) containing numerical features from test data
    :param list y_test: a list of gold labels from test data
    :param pandas.Dataframe df_test: the test dataframe which will be used to store the candidate predictions
    :param str step: 'candidates' for candidate detection, 'roles' for role labelling
    """
    # vectorize categorical features
    dv = DictVectorizer()
    X_train_cat_vectorized = dv.fit_transform(X_train_cat_dicts)
    print('Categorical training features vectorized.')
    X_test_cat_vectorized = dv.transform(X_test_cat_dicts)
    print('Categorical test features vectorized.')
    
    # vectorize numerical features
    X_train_num_list = []
    for numerical_dictionary in X_train_num_dicts:
        X_train_num_list.append([v for v in numerical_dictionary.values()])

    X_test_num_list = []
    for numerical_dictionary in X_test_num_dicts:
        X_test_num_list.append([v for v in numerical_dictionary.values()])
    
    # concatenate categorical and numerical features
    X_train_vectorized = concatenate_categorical_and_numerical_feature_vectors(X_train_cat_vectorized,
                                                                               X_train_num_list)
    print('Numerical and categorical training feature vectors concatenated.')
    X_test_vectorized = concatenate_categorical_and_numerical_feature_vectors(X_test_cat_vectorized,
                                                                              X_test_num_list)
    print('Numerical and categorical test feature vectors concatenated.')
    
    # instantiate the model and fit it to the concatenated features and the gold labels of the training data
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_vectorized, y_train)
    print('LogReg model trained.')

    # write predictions to file, so we can extract features from the predicted candidates for the next step
    if step == 'candidates':
        y_pred = model.predict(X_test_vectorized)
        print('Candidate predictions made.')
        df_test['candidate_prediction'] = y_pred
        df_test.to_csv(f'Data/test_data_with_candidate_predictions.tsv', sep='\t', mode='w', header=True,
                       index=False)
        print('Predicted candidates for test data stored.')
    elif step == 'roles':  # or get the probability distributions for the semantic role labels
        y_pred = model.predict_proba(X_test_vectorized)
        print('Semantic role redictions made.')
        df_test['predicted_semantic_role'] = y_pred
        print(y_pred)
        df_test.to_csv(f'Data/test_data_with_role_predictions.tsv', sep='\t', mode='w', header=True,
                       index=False)
    else:
        raise ValueError
    
    # print classification report for the gold labels vs. the predicted labels of the test data
    print(classification_report(y_test, y_pred, digits=3))
