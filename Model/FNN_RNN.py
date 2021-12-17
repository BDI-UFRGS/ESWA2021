import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np

def run(X_train, y_train, X_test, n_outputs, epochs = 10):
    comment_train, comment_test = X_train.iloc[:, 1], X_test.iloc[:, 1]
    word_vectors_train, word_vectors_test = X_train.iloc[:, 2:], X_test.iloc[:, 2:]
    
    m = model(comment_train, n_outputs)
    m = fit(m, [comment_train, word_vectors_train], y_train, epochs=epochs)

    return predict(m, [comment_test, word_vectors_test])

def model(comment, n_outputs):
    encoder_train = preprocessing.TextVectorization(output_mode="int")
    encoder_train.adapt(np.asarray(comment))

    rnn_input = tf.keras.Input(shape=(1,), dtype="string")
    rnn_bi_ltsm = encoder_train(rnn_input)
    rnn_bi_ltsm = layers.Embedding(input_dim=len(
        encoder_train.get_vocabulary()), output_dim=64)(rnn_bi_ltsm)

    rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(64))(rnn_bi_ltsm)
    rnn_output = layers.Dense(64, activation='relu')(rnn_bi_ltsm)

    fnn_input = tf.keras.Input(shape=(300,))
    fnn = layers.Dense(300, activation='relu')(fnn_input)
    fnn = layers.Dense(300, activation='relu')(fnn)
    fnn_output = layers.Dense(64, activation='relu')(fnn)

    merge_layer = tf.keras.layers.Concatenate()([rnn_output, fnn_output])

    out = layers.Flatten()(merge_layer)

    out = layers.Dense(128, activation='relu')(out)
    out = layers.Dense(128, activation='relu')(out)

    out = layers.Dense(n_outputs, activation='softmax')(out)

    model = tf.keras.models.Model([rnn_input, fnn_input], out)

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def fit(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model


def predict(model, X_test):
    predictions = model.predict(X_test)

    return predictions
