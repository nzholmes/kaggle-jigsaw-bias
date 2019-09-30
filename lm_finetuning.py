import pandas as pd
import torch
from torch import nn
from torch.utils.data import RandomSampler
import numpy as np
from tqdm import tqdm
from config.base import (
    TOXICITY_COLUMN, IDENTITY_COLUMNS, AUX_TOXICITY_COLUMNS, TEXT_COLUMN, OUTPUT_DIR, Y_COLUMNS,
    TRAIN_DATA, TEST_DATA,
    TRAIN_PREPROCESSED_DATA, TEST_PREPROCESSED_DATA)

from src.utils import seed_torch, df_parallelize_run, read_json
from src.weights import training_weights

from src.preprocessing import preprocess_final

from src.pytorch_pretrained_bert import BertTokenizer, BertAdam
from src.pytorch_pretrained_bert.optimization import SCHEDULES
from src.lm_models.models import BertForSequenceClassification_LayerConcat4, \
    BertForSequenceClassification_StaticWeight4, BertForSequenceClassification_SecondtoLast
from src.lm_models.data import LenMatchBatchSampler, trim_tensors
from src.lm_models.tokenize import convert_lines
from src.lm_models.utils import model_predict

from apex.optimizers import FusedAdam
from apex import amp


def main():

    # config settings

    config = read_json("../config/bert_base_uncased.json")
    SEED = config.get('seed', 1029)
    seed_torch(seed=SEED)
    lm_model_name = config.get("lm_model_name", "bert_base_uncased")
    bn_sn_ratio = config.get('bn_sn_ratio', 4)
    maxlen = config.get('maxlen', 220)
    n_target = len(Y_COLUMNS)+1
    batch_size = config.get('batch_size', 32)
    model_path = config.get('model_path', './input/pretrained_model')
    epochs = config.get("epochs", 2)
    accumulation_steps = config.get("accumulation_steps")
    warmup_proportion = config.get("warmup_proportion")
    scheduler_type = config.get("scheduler_type")
    is_apex_optimizer = config.get("is_apex_optimizer")
    model_type = config.get("model_type", "last_concat4")
    device = config.get("device")
    lr = config.get('lr', 2e-5)

    if model_type == "last_concat4":
        Model = BertForSequenceClassification_LayerConcat4
    elif model_type == "second2last":
        Model = BertForSequenceClassification_SecondtoLast
    elif model_type == "staticweight4":
        Model = BertForSequenceClassification_StaticWeight4

    BERT_OUTPUT = OUTPUT_DIR / "BERT_BASE_UNCASED_OUTPUT"
    BERT_OUTPUT.mkdir(exist_ok=True)

    # preprocessing
    print("START PREPROCESSING")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    num_train_data = train_df.shape[0]

    txt_df = pd.concat([train_df[TEXT_COLUMN], test_df[TEXT_COLUMN]], axis=0)
    preprocessed_txt = df_parallelize_run(txt_df[TEXT_COLUMN].fillna(''), preprocess_final)
    txt_df[TEXT_COLUMN] = preprocessed_txt
    train_df[TEXT_COLUMN] = preprocessed_txt[:num_train_data]
    test_df[TEXT_COLUMN] = preprocessed_txt[train_df.shape[0]:]

    train_df.to_csv(TRAIN_PREPROCESSED_DATA)
    test_df.to_csv(TEST_PREPROCESSED_DATA)

    tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=None, do_lower_case=True)
    bert_text, bert_matrix = convert_lines(txt_df[TEXT_COLUMN].fillna("DUMMY_VALUE"), maxlen, tokenizer, True)
    txt_df['comment_text_bert_uncase'] = bert_text
    txt_df.to_csv(BERT_OUTPUT/"bert_uncase_txt.csv.gzip", index=False, compression='gzip')
    np.savez_compressed(BERT_OUTPUT/"bert_maxlen{}.npz".format(maxlen), bert_matrix=bert_matrix)

    x_train = bert_matrix[:num_train_data]
    x_test = bert_matrix[num_train_data:]

    y_train = train_df[TOXICITY_COLUMN].values
    weights = training_weights(train_df, IDENTITY_COLUMNS, bn_sn_ratio)
    y_train = np.vstack([y_train, weights]).T.astype(np.float32)
    y_identity = train_df[IDENTITY_COLUMNS].fillna(0).values
    y_aux = train_df[[TOXICITY_COLUMN] + AUX_TOXICITY_COLUMNS].values
    y_train_total = np.hstack([y_train, y_aux, y_identity.values]).astype(np.float32)
    print("FINISH PREPROCESSING")

    # training
    print("START TRAINING")

    loss_weight = 1.0/weights.mean()
    def custom_loss(y_pred, y_true):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss1 = nn.BCEWithLogitsLoss(weight=y_true[:, 1:2])(y_pred[:, 0:1], y_true[:, 0:1])
        bce_loss2 = nn.BCEWithLogitsLoss()(y_pred[:, 1:], y_true[:, 2:n_target])
        return (bce_loss1 * loss_weight) + bce_loss2

    loss_fn = custom_loss

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.long),
                                                   torch.tensor(y_train_total[:, :n_target], dtype=torch.float32))

    # model setup
    model = Model.from_pretrained(model_path, cache_dir=None, num_labels=len(Y_COLUMNS))
    model.zero_grad()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(len(train_dataset) / batch_size / accumulation_steps) * epochs

    if not is_apex_optimizer:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lr,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    else:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        warmup_scheduler = SCHEDULES[scheduler_type]

    random_sampler = RandomSampler(train_dataset)
    len_sampler = LenMatchBatchSampler(random_sampler, batch_size=batch_size, drop_last=False)

    # use mixed precision training
    if is_apex_optimizer:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                          loss_scale="dynamic", verbosity=0)
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic", verbosity=0)
    model = model.train()

    global_step = 0
    for epoch in range(epochs):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=len_sampler)

        avg_loss = 0.
        avg_accuracy = 0.

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for i, batch in tk0:
            # optimizer.zero_grad()
            tsrs = trim_tensors(batch)
            x_batch, y_batch = tuple(t.to(device) for t in tsrs)

            y_pred = model(x_batch, attention_mask=x_batch > 0, labels=None)

            loss = loss_fn(y_pred, y_batch)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
                if is_apex_optimizer:
                    lr_this_step = lr * warmup_scheduler(global_step / num_train_optimization_steps, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()
                global_step += 1

            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(
                ((torch.sigmoid(y_pred[:, 0]) > 0.5)
                 == (y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item() / len(train_loader)
            tk0.set_postfix(loss=loss.item(), acc=avg_accuracy)

    torch.save(model.state_dict(), BERT_OUTPUT/"bert_pytorch_finetune_{}.bin".format(lm_model_name))

    print("FINISH TRAINING")

    # test
    test_preds = model_predict(model, x_test)

    # submission
    print("SUBMISSION GENERATION")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_preds
    })
    submission.to_csv(BERT_OUTPUT/"bert_submission.csv", index=False)
