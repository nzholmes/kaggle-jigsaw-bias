import os
import random
import time
import json
import psutil
import pandas as pd
import numpy as np
import torch
from multiprocessing import Pool
from contextlib import contextmanager

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def df_parallelize_run(text_list, func, num_partitions=4):
    num_cores = psutil.cpu_count()
    df_split = np.array_split(text_list, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def save_data2json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d


def cut_length(data, mask):
    max_length = data.shape[1]
    transposed = torch.transpose(data, 1, 0)
    res = (transposed == mask).all(1)
    for i, r in enumerate(res):
        if r == 0:
            break
    data = data[:, -(max_length - i):]
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))