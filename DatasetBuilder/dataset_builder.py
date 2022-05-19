import io
import numpy as np
import csv
from gensim import models

fasttext_crawl_300d_path = 'vector_models/fasttext/crawl-300d-2M.vec'
fasttext_crawl_300d_subword_path = 'vector_models/fasttext/crawl-300d-2M-subword.vec'
fasttext_wiki_news_300d_path = 'vector_models/fasttext/wiki-news-300d-1M.vec'
fasttext_wiki_news_300d_subword_path = 'vector_models/fasttext/wiki-news-300d-1M-subword.vec'
glove_6b_300d_path = 'vector_models/glove/glove.6B.300d.txt'
glove_42b_300d_path = 'vector_models/glove/glove.42B.300d.txt'
glove_840b_300d_path = 'vector_models/glove/glove.840B.300d.txt'
word2vec_path = 'vector_models/word2vec/GoogleNews-vectors-negative300.bin'


def load_word2vec_model(fname, binary=False):
    data = models.KeyedVectors.load_word2vec_format(fname, binary=binary)
    return data

def load_model(fname):
    print('Loading Model: %s' % fname)
    data = {}
    with open(fname,'r', errors='ignore', encoding='utf-8') as f:
        for line in f:
            split_line = line.split(' ')
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            data[word] = embedding
        f.close()
    print('Loading %s finished' % fname)

    return data

def read_dataset(fdataset):
    entities = list()

    with open(fdataset, newline='') as f:
        spamreader = csv.reader(f, delimiter=';', quotechar='|')
        for row in spamreader:
            entity = dict()
            entity['class'] = row[0]
            entity['term'] = row[1]
            entity['comment'] = row[2]
            entities.append(entity)

    return entities


def create_embeddings_dataset(dataset_entities, embeddings_model, name):
    print('Creating dataset: %s' % name)
    f = open('%s.csv' % name, 'w+')

    for entity in dataset_entities:
        if entity['term'] in embeddings_model:
            f.write('%s;%s;%s;%s\n' % (entity['class'], entity['term'], entity['comment'], ';'.join(str(v) for v in embeddings_model[entity['term']])))
        else:
            entity_words = entity['term'].split(' ')
            if len(entity_words) > 1 and all([entity_word in embeddings_model for entity_word in entity_words]): 
                print('Sim %s' % entity['term'])

                vec = embeddings_model[entity_words[0]]
                for entity_word in entity_words[1:]:
                    vec += embeddings_model[entity_word]

                vec = vec / len(entity_words)
                f.write('%s;%s;%s;%s\n' % (entity['class'], entity['term'], entity['comment'], ';'.join(str(v) for v in vec)))

            else:
                print('Nao %s' % entity['term'])
                



    f.close()
    print('Dataset %s created' % name)


def build(fdataset):

    dataset_entities = read_dataset(fdataset)

    # fasttext_crawl_300d_model = load_model(fasttext_crawl_300d_path)
    # create_embeddings_dataset(dataset_entities, fasttext_crawl_300d_model, 'fasttext_crawl_300d')    
    # del(fasttext_crawl_300d_model)

    # fasttext_crawl_300d_subword_model = load_model(fasttext_crawl_300d_subword_path)
    # create_embeddings_dataset(dataset_entities, fasttext_crawl_300d_subword_model, 'fasttext_crawl_300d_subword')    
    # del(fasttext_crawl_300d_subword_model)

    # fasttext_wiki_news_300d_model = load_model(fasttext_wiki_news_300d_path)
    # create_embeddings_dataset(dataset_entities, fasttext_wiki_news_300d_model, 'fasttext_wiki_news_300d')    
    # del(fasttext_wiki_news_300d_model)

    # fasttext_wiki_news_300d_subword_model = load_model(fasttext_wiki_news_300d_subword_path)
    # create_embeddings_dataset(dataset_entities, fasttext_wiki_news_300d_subword_model, 'fasttext_wiki_news_300d_subword')    
    # del(fasttext_wiki_news_300d_subword_model)

    glove_6b_300d_model = load_model(glove_6b_300d_path)
    create_embeddings_dataset(dataset_entities, glove_6b_300d_model, 'glove_6b_300d')    
    del(glove_6b_300d_model)

    # glove_42b_300d_model = load_model(glove_42b_300d_path)
    # create_embeddings_dataset(dataset_entities, glove_42b_300d_model, 'glove_42b_300d')    
    # del(glove_42b_300d_model)

    # glove_840b_300d_model = load_model(glove_840b_300d_path)
    # create_embeddings_dataset(dataset_entities, glove_840b_300d_model, 'glove_840b_300d')    
    # del(glove_840b_300d_model)

    # word2vec_model = load_word2vec_model(word2vec_path, binary=True)
    # create_embeddings_dataset(dataset_entities, word2vec_model, 'word2vec')
    # del(word2vec_model)
