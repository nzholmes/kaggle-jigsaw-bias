import numpy as np
from src.pytorch_pretrained_bert import BertConfig
import torch
import pandas as pd
from tqdm import tqdm
from .data import trim_tensors

def get_model_prediction(model_path, model_config_path, X_test, Model=None):

    # bert_config = BertConfig(os.path.join(model_path, 'bert_config.json'))
    bert_config = BertConfig(model_config_path)

    model = Model(bert_config, num_labels=7)

    # model_path = os.path.join(model_path, "bert_pytorch.bin")
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")
    for param in model.parameters():
        param.requires_grad = False

    test_pred = model_predict(model, X_test)

    return test_pred

def model_predict(model, X_test):
    model.eval()

    sort_len = pd.Series(np.sum(X_test != 0, 1)).sort_values()
    sort_len_index = sort_len.index.tolist()
    X_test = X_test[sort_len_index]

    test_preds = np.zeros((len(X_test)))
    test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    tk0 = tqdm(test_loader)
    for i, (x_batch,) in enumerate(tk0):
        x_batch = trim_tensors(x_batch)
        pred = model(x_batch.to("cuda"), attention_mask=(x_batch > 0).to("cuda"), labels=None)
        test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

    test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()

    test_pred = pd.Series(test_pred, index=sort_len_index).sort_index().values

    return test_pred