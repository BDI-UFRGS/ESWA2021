import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing

np.random.seed(1)


def remove_non_elegible(df, class_index, n_classes):
    value_counts = df[df.columns[class_index]].value_counts()
    elegible_indexes = value_counts.index[0:n_classes]

    non_duplicated_df = pd.DataFrame()
    duplicated_df = pd.DataFrame()

    i = 0
    for index in elegible_indexes:
        item = df[df[df.columns[class_index]] == index]

        non_duplicated = item[~item.duplicated(item.columns[2], keep='first')]
        duplicated = item[item.duplicated(item.columns[2], keep='first')]

        duplicated_df = pd.concat([duplicated_df, duplicated])
        non_duplicated_df = pd.concat([non_duplicated_df, non_duplicated])
        
    duplicated_df = pd.concat([duplicated_df, non_duplicated_df[non_duplicated_df.duplicated(non_duplicated_df.columns[2], keep=False)]])

    non_duplicated_df = non_duplicated_df[~non_duplicated_df.duplicated(non_duplicated_df.columns[2], keep=False)]
    
    print("Tamanho final duplicados: %s" % len(duplicated_df))

    print("Tamanho final não duplicados: %s" % len(non_duplicated_df))

    return non_duplicated_df, duplicated_df

def get_dataset(path, delimiter, class_index, n_classes):
    
    df = pd.read_csv(path, delimiter=delimiter)

    df=df.dropna()
    
    non_duplicated_df, duplicated_df = remove_non_elegible(df, class_index, n_classes)

    return non_duplicated_df, duplicated_df

def show_accuracy_graph(y_test_non_category, predictions_non_category, class_names):
    conf_mat = confusion_matrix(y_test_non_category, predictions_non_category)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] 

    df_cm = pd.DataFrame(conf_mat, index = [i for i in class_names], columns = [i for i in class_names])
    plt.figure(figsize = (10,6))
    sn.heatmap(df_cm, annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.show()

def get_results(y_test_non_category, predictions_non_category):
    f1_micro = f1_score(y_test_non_category, predictions_non_category, average='micro', labels=np.unique(predictions_non_category))
    f1_macro = f1_score(y_test_non_category, predictions_non_category, average='macro',labels=np.unique(predictions_non_category))
    f1_weighted = f1_score(y_test_non_category, predictions_non_category, average='weighted',labels=np.unique(predictions_non_category))
    precision_micro = precision_score(y_test_non_category, predictions_non_category, average='micro',labels=np.unique(predictions_non_category))
    precision_macro = precision_score(y_test_non_category, predictions_non_category, average='macro',labels=np.unique(predictions_non_category))
    precision_weighted = precision_score(y_test_non_category, predictions_non_category, average='weighted',labels=np.unique(predictions_non_category))
    recall_micro = recall_score(y_test_non_category, predictions_non_category, average='micro',labels=np.unique(predictions_non_category))
    recall_macro = recall_score(y_test_non_category, predictions_non_category, average='macro',labels=np.unique(predictions_non_category))
    recall_weighted = recall_score(y_test_non_category, predictions_non_category, average='weighted',labels=np.unique(predictions_non_category))

    return [f1_micro, f1_macro, f1_weighted, precision_micro,precision_macro,precision_weighted,recall_micro,recall_macro,recall_weighted]

def get_minority_length(df):
    value_counts = df[df.columns[0]].value_counts()
    return min([df[df[df.columns[0]] == index].shape[0] for index in value_counts.index])

def get_minority_length(df):
    value_counts = df[df.columns[0]].value_counts()
    return min([df[df[df.columns[0]] == index].shape[0] for index in value_counts.index])

def downsample(X_train_fold, y_train_fold):
    df = pd.concat([y_train_fold, X_train_fold], axis=1)
    value_counts = df[df.columns[0]].value_counts()

    min_length = get_minority_length(df)

    test_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for index in value_counts.index:
        item = df[df[df.columns[0]] == index]
        item_shape = item.shape[0]
        all_indexes = pd.Index(np.arange(item_shape))
        chosen_idx = np.random.choice(item_shape, replace=False, size=min_length) 
        ignored_indexes = all_indexes.difference(chosen_idx)

        test_df = pd.concat([test_df, item.iloc[chosen_idx]], axis=0)
        val_df = pd.concat([val_df, item.iloc[ignored_indexes]], axis=0)        
        
    return test_df.iloc[:, 1:], val_df.iloc[:, 1:], test_df.iloc[:, 0], val_df.iloc[:, 0]

def remove_non_elegible2(df, class_index, n_classes):
    value_counts = df[df.columns[class_index]].value_counts()
    elegible_indexes = value_counts.index[0:n_classes]

    non_duplicated_df = pd.DataFrame()

    i = 0
    for index in elegible_indexes:
        item = df[df[df.columns[class_index]] == index]

        non_duplicated_df = pd.concat([non_duplicated_df, item])

    return non_duplicated_df

    

def get_dataset2(path, delimiter, class_index, n_classes):
    
    df = pd.read_csv(path, delimiter=delimiter)

    df=df.dropna()
    
    non_duplicated_df = remove_non_elegible2(df, class_index, n_classes)

    return non_duplicated_df

def p2_train_test(path, name, class_index, n_samples, exp):
    
    num_folds = 10
    non_duplicated_df = get_dataset2(path, ';', class_index, n_samples)
    print("Tamanho final não duplicados: %s" % len(non_duplicated_df))

    skf = StratifiedKFold(n_splits=num_folds)

    X = non_duplicated_df.iloc[:, 2:]
    y = non_duplicated_df.iloc[:, class_index]
    y = y.astype(str)

    class_names = np.unique(y)
    n_outputs = len(class_names)
    final_results = np.zeros(9)
    i = 0
    for train_index, val_index in skf.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_index], y.iloc[train_index]
        X_val_fold, y_val_fold = X.iloc[val_index], y.iloc[val_index]

        # X_train_fold, y_train_fold = upsample(X_train_fold, y_train_fold)

        X_train_fold, X_val_fold_extra, y_train_fold, y_val_fold_extra = downsample(X_train_fold, y_train_fold)

        X_val_fold = pd.concat([X_val_fold, X_val_fold_extra])
        y_val_fold = pd.concat([y_val_fold, y_val_fold_extra])

        print("Train Size:", len(X_train_fold))
        print("Test Size:", len(X_val_fold))

        comment_train, comment_val = X_train_fold.iloc[:, 1], X_val_fold.iloc[:, 1]
        word_vectors_train, word_vectors_val = X_train_fold.iloc[:, 2:], X_val_fold.iloc[:, 2:]

        y_train = pd.get_dummies(y_train_fold)
        y_val = pd.get_dummies(y_val_fold)

        encoder_train = preprocessing.TextVectorization(output_mode="int")
        encoder_train.adapt(np.asarray(comment_train))

        rnn_input = tf.keras.Input(shape=(1,), dtype="string")
        rnn_bi_ltsm = encoder_train(rnn_input)
        rnn_bi_ltsm = layers.Embedding(input_dim=len(encoder_train.get_vocabulary()), output_dim=64)(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(64))(rnn_bi_ltsm)
        rnn_output = layers.Dense(64, activation='relu')(rnn_bi_ltsm)

        fnn_input = tf.keras.Input(shape=(300,))
        fnn = layers.Flatten()(fnn_input)
        fnn_output = layers.Dense(64, activation='relu')(fnn)
        
        merge_layer = tf.keras.layers.Concatenate()([rnn_output, fnn_output])

        out = layers.Dense(64, activation='relu')(merge_layer)
        
        out = layers.Dense(n_outputs, activation='softmax')(out)

        model = tf.keras.models.Model([rnn_input, fnn_input], out)

        model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        history = model.fit([comment_train, word_vectors_train], y_train, epochs=10, verbose=2)

        predictions = model.predict([comment_val, word_vectors_val])

        y_test_non_category = [np.argmax(t, axis=0) for t in np.asarray(y_val)]
        predictions_non_category = [np.argmax(t) for t in predictions]
        print(len(y_test_non_category))
        print(len(predictions_non_category))
        dif = set(y_test_non_category) - set(predictions_non_category)
        print(dif)
    
        results = get_results(y_test_non_category, predictions_non_category)
        file_output = open(exp + '.txt', 'a')
        file_output.write("p2;"+ name + ";" + ';'.join([str(x) for x in results]) + "\n")
        file_output.close()

        print(';'.join([str(x) for x in results]))
        if len(dif) == 0:       
            final_results += results
            i = i + 1  
        show_accuracy_graph(y_test_non_category, predictions_non_category, class_names)

    final_results /= i
    file_output = open(exp + '.txt', 'a')
    file_output.write("p2-final;"+ name + ";" + ';'.join([str(x) for x in final_results]) + "\n")
    file_output.close()
    print(';'.join([str(x) for x in final_results]))

def my_train_test(path, name, class_index, n_samples, exp):
    
    num_folds = 10
    non_duplicated_df, duplicated_df = get_dataset(path, ';', class_index, n_samples)
    print("Tamanho final não duplicados: %s" % len(non_duplicated_df))
    print("Tamanho final duplicados: %s" % len(duplicated_df))

    skf = StratifiedKFold(n_splits=num_folds)

    X = non_duplicated_df.iloc[:, 2:]
    y = non_duplicated_df.iloc[:, class_index]
    y = y.astype(str)

    X2 = duplicated_df.iloc[:, 2:]
    y2 = duplicated_df.iloc[:, class_index]
    y2 = y2.astype(str)

    class_names = np.unique(y)
    n_outputs = len(class_names)
    final_results = np.zeros(9)
    i = 0
    for train_index, val_index in skf.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_index], y.iloc[train_index]
        X_val_fold, y_val_fold = X.iloc[val_index], y.iloc[val_index]

        # X_train_fold, y_train_fold = upsample(X_train_fold, y_train_fold)

        X_train_fold, X_val_fold_extra, y_train_fold, y_val_fold_extra = downsample(X_train_fold, y_train_fold)

        X_val_fold = pd.concat([X_val_fold, X_val_fold_extra])
        y_val_fold = pd.concat([y_val_fold, y_val_fold_extra])

        comment_train, comment_val = X_train_fold.iloc[:, 1], X_val_fold.iloc[:, 1]
        word_vectors_train, word_vectors_val = X_train_fold.iloc[:, 2:], X_val_fold.iloc[:, 2:]

        y_train = pd.get_dummies(y_train_fold)
        y_val = pd.get_dummies(y_val_fold)

        encoder_train = preprocessing.TextVectorization(output_mode="int")
        encoder_train.adapt(np.asarray(comment_train))

        rnn_input = tf.keras.Input(shape=(1,), dtype="string")
        rnn_bi_ltsm = encoder_train(rnn_input)
        rnn_bi_ltsm = layers.Embedding(input_dim=len(encoder_train.get_vocabulary()), output_dim=64)(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(64))(rnn_bi_ltsm)
        rnn_output = layers.Dense(64, activation='relu')(rnn_bi_ltsm)

        fnn_input = tf.keras.Input(shape=(300,))
        fnn = layers.Flatten()(fnn_input)
        fnn_output = layers.Dense(64, activation='relu')(fnn)
        
        merge_layer = tf.keras.layers.Concatenate()([rnn_output, fnn_output])

        out = layers.Dense(64, activation='relu')(merge_layer)
        
        out = layers.Dense(n_outputs, activation='softmax')(out)

        model = tf.keras.models.Model([rnn_input, fnn_input], out)

        model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        history = model.fit([comment_train, word_vectors_train], y_train, epochs=10, verbose=2)

        predictions = model.predict([comment_val, word_vectors_val])

        y_test_non_category = [np.argmax(t, axis=0) for t in np.asarray(y_val)]
        predictions_non_category = [np.argmax(t) for t in predictions]
        print(len(y_test_non_category))
        print(len(predictions_non_category))
        dif = set(y_test_non_category) - set(predictions_non_category)
        print(dif)
    
        results = get_results(y_test_non_category, predictions_non_category)
        file_output = open(exp + '.txt', 'a')
        file_output.write("p2;"+ name + ";" + ';'.join([str(x) for x in results]) + "\n")
        file_output.close()

        print(';'.join([str(x) for x in results]))
        if len(dif) == 0:       
            final_results += results
            i = i + 1  
        show_accuracy_graph(y_test_non_category, predictions_non_category, class_names)

    final_results /= i
    file_output = open(exp + '.txt', 'a')
    file_output.write("p2-final;"+ name + ";" + ';'.join([str(x) for x in final_results]) + "\n")
    file_output.close()
    print(';'.join([str(x) for x in final_results]))

my_train_test('dataset/fasttext/crawl-300d-2M/particular.csv', 'crawl-300d-2M', 0, 2, 'exp1')
