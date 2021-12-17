import tensorflow as tf
from tensorflow.keras import layers

def run(X_train, y_train, X_test, n_outputs, epochs = 10):
    word_vectors_train, word_vectors_test = X_train.iloc[:, 2:], X_test.iloc[:, 2:]
    
    m = model(n_outputs)
    m = fit(m, word_vectors_train, y_train, epochs=epochs)

    return predict(m, word_vectors_test)

def model(n_outputs):

    fnn_input = tf.keras.Input(shape=(300,))
    fnn = layers.Dense(300, activation='relu')(fnn_input)
    fnn = layers.Dense(300, activation='relu')(fnn)
    fnn_output = layers.Dense(64, activation='relu')(fnn)

    out = layers.Dense(n_outputs, activation='softmax')(fnn_output)


    model = tf.keras.models.Model(fnn_input, out)

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

