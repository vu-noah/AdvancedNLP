from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_logreg(X_train_cat, X_train_num, y_train, X_test_cat, X_test_num, y_test):
    '''
    Run Logistic Regression model.   
    
    :param X_train_cat: a dictionary containing categorical features from training data
    :param X_train_num: a list (of lists) containing numerical features from training data
    :param y_train: a list of gold labels from training data 
    :param X_test_cat: a dictionary containing categorical features from test data
    :param X_test_num: a list (of lists) containing numerical features from test data
    :param y_test: a list of gold labels from test data 
    '''
    # vectorize categorical features
    dv = DictVectorizer()
    X_train_cat_vectorized = dv.fit_transform(X_train_cat)
    X_test_cat_vectorized = dv.transform(X_test_cat)
    
    # vactorize numerical features 
    X_train_num_vectorized = numpy.array(X_train_num)
    X_test_num_vectorized = numpy.array(X_test_num)
    
    # concatenate (TO DO)
    X_train_vectorized = 
    X_test_vectorized = 
    
    # run the model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)

    print(classification_report(y_test, y_pred, digits = 3))