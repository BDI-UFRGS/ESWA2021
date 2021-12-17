from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def run(X_train, y_train, X_test):
    word_vectors_train, word_vectors_test = X_train.iloc[:, 2:], X_test.iloc[:, 2:]
    y_train = [np.argmax(t, axis=0) for t in np.asarray(y_train)]
    m = model()
    m = fit(m, word_vectors_train, y_train)

    return predict(m, word_vectors_test)

def model():
    return svm.SVC(probability=True, verbose=1)


def fit(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    predictions = model.predict_proba(X_test)

    return predictions

