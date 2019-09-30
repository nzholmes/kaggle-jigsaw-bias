import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer('english')


def load_train_test(data_path):
    a = pd.read_csv(data_path + "/train_df.csv", nrows=5).dtypes.to_dict()
    columns_types = {'int64': 'int32', 'float64': 'float32'}
    usecols = {c: columns_types[v.name] for c, v in a.items() if v.name != 'object'}
    usecols.update({c: v.name for c, v in a.items() if v.name == 'object'})

    train_df = pd.read_csv(data_path + "/train_df.csv", usecols=list(usecols.keys()), dtype=usecols)
    test_df = pd.read_csv(data_path + "/test_df.csv")

    return train_df, test_df

def load_embedding(embedding_path, word_dict, max_features, embed_size=300):
    """
    Load embedding for each token in word_dict.
    To determine whether a token is in the embedding, the lower, upper, captial form and stemming
    are used.
    :param embedding_path: str, the path to embedding
    :param word_dict: dict, dict of token and index
    :param max_features: int, maximum number of features for embedding
    :param embed_size: int, embedding dimension
    :return: embedding_matrix: np.array with dimension of (min(max_features, len(word_dict) + 1), embed_size)
    """
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.strip().split(' ')) for o in open(embedding_path))

    num_pattern = re.compile(r'(\d+)\.(\d+)')
    num_embedding = np.mean([embeddings_index[w] for w in embeddings_index if num_pattern.search(w) is not None],
                            axis=0)

    url_pattern = re.compile(
        r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}\
|[a-z0-9%])\|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\)))'
    )
    url_embedding = np.mean([embeddings_index[w] for w in embeddings_index if url_pattern.search(w) is not None],
                            axis=0)

    # len(word_dict)+1 because of padded token
    nb_words = min(max_features, len(word_dict) + 1)

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    for key, index in word_dict.items():

        if index >= max_features:
            continue

        if key.startswith("NUM_"):
            embedding_matrix[word_dict[key]] = num_embedding
            continue

        word = key if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower() if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper() if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize() if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key) if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key) if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key) if not key.startswith('URL_') else key.replace('URL_', '')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        if key.startswith('URL_'):
            embedding_matrix[word_dict[key]] = url_embedding
        else:
            embedding_matrix[word_dict[key]] = unknown_vector

    return embedding_matrix
