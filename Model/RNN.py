import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import time

def run(X_train, y_train, X_test, n_outputs, epochs = 10, proba=False):
    print('Running RNN finished...')

    comment_train, comment_test = X_train.iloc[:, 1], X_test.iloc[:, 1]
    
    m = model(comment_train, n_outputs)
    ini_time = time.time_ns()
    m = fit(m, comment_train, y_train, epochs=epochs)
    end_time = time.time_ns()

    if proba:
        predictions = predict_proba(m, comment_test)
    else:
        predictions = predict(m, comment_test)
        
    print('RNN finished.')
    return predictions, end_time-ini_time

def model(comment, n_outputs):
    encoder_train = preprocessing.TextVectorization(output_mode="int")
    encoder_train.adapt(np.asarray(comment))

    rnn_input = tf.keras.Input(shape=(1,), dtype="string")
    rnn_bi_ltsm = encoder_train(rnn_input)
    rnn_bi_ltsm = layers.Embedding(input_dim=len(
        encoder_train.get_vocabulary()), output_dim=64)(rnn_bi_ltsm)

    rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(64))(rnn_bi_ltsm)
    rnn_output = layers.Dense(64, activation='relu')(rnn_bi_ltsm)

    out = layers.Dense(128, activation='relu')(rnn_output)

    out = layers.Dense(n_outputs, activation='softmax')(out)

    model = tf.keras.models.Model(rnn_input, out)

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
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
