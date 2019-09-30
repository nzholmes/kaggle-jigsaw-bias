import tqdm

def text2index(text_list, max_features=None):
    """
    Text tokenization
    :param text_list: list of preprocessed text
    :param max_features: int, maximum number of tokens in vocabulary
    :return: word2index: dict, vocabulary in text;
             all_sequences: list of index of tokens
    """
    word2index = dict()
    index = 1  # index start with 1
    all_sequences = list()

    for s in tqdm(text_list):
        word_seq = list()
        for token in s.split():
            token = "NUM_##" if token.startswith('NUM_') else token

            if token not in word2index:
                if max_features is not None and index >= max_features - 1:
                    token = '<UNK>'
                    word2index[token] = max_features - 1
                else:
                    word2index[token] = index
                    index += 1
            word_seq.append(word2index[token])
        all_sequences.append(word_seq)
    return word2index, all_sequences
