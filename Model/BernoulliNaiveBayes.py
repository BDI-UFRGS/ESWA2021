from sklearn.naive_bayes import BernoulliNB
import numpy as np
import time
def run(X_train, y_train, X_test, proba=False):
    print('Running Bernoulli Naive Bayes...')
    word_vectors_train, word_vectors_test = X_train.iloc[:, 2:], X_test.iloc[:, 2:]
    y_train = [np.argmax(t, axis=0) for t in np.asarray(y_train)]
    m = model()

    ini_time = time.time_ns()
    m = fit(m, word_vectors_train, y_train)
    end_time = time.time_ns()

    if proba:
        predictions = predict_proba(m, word_vectors_test)
    else:
        predictions = predict(m, word_vectors_test)
    
    print('Bernoulli Naive Bayes finished.')

    return predictions, end_time-ini_time

def model():
    return BernoulliNB()


def fit(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X_test):
    predictions = model.predict_proba(X_test)

    return predictions

def predict(model, X_test):
    predictions = model.predict(X_test)

    return predictions