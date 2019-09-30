import numpy as np
from tqdm import tqdm


def convert_lines(text_list, max_seq_length, tokenizer, keep_bert_text=False):
    max_seq_length -= 2
    all_tokens = []
    all_text = []
    longer = 0
    for text in tqdm(text_list):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
        if keep_bert_text:
            all_text.append(" ".join(tokens_a))
    if keep_bert_text:
        return all_text, np.array(all_tokens, dtype=np.int32)
    else:
        return np.array(all_tokens, dtype=np.int32)