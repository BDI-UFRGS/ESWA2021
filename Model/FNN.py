import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time

def run(X_train, y_train, X_test, n_outputs, epochs = 10, proba=False):
    print('Running FNN...')

    word_vectors_train, word_vectors_test = X_train.iloc[:, 2:], X_test.iloc[:, 2:]
    
    m = model(n_outputs)
    ini_time = time.time_ns()
    m = fit(m, word_vectors_train, y_train, epochs=epochs)
    end_time = time.time_ns()

    if proba:
        predictions = predict_proba(m, word_vectors_test)
    else:
        predictions = predict(m, word_vectors_test)
        
    print('FNN finished.')
    return predictions, end_time-ini_time

def model(n_outputs):

    fnn_input = tf.keras.Input(shape=(300,))
    # fnn = layers.Dense(300, activation='relu')(fnn_input)
    # fnn = layers.Dense(300, activation='relu')(fnn)
    fnn_output = layers.Dense(64, activation='relu')(fnn_input)

    out = layers.Dense(n_outputs, activation='softmax')(fnn_output)


    model = tf.keras.models.Model(fnn_input, out)

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def fit(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model


def predict_proba(model, X_test):
    predictions = model.predict(X_test)

    return predictions

def predict(model, X_test):
    predictions = model.predict(X_test)
    return [np.argmax(t, axis=0) for t in np.asarray(predictions)]