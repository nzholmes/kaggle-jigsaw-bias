import pandas as pd
import torch
from torch import nn
import gc
import numpy as np
import time
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from config.base import (
    TOXICITY_COLUMN, IDENTITY_COLUMNS, AUX_TOXICITY_COLUMNS, TEXT_COLUMN, OUTPUT_DIR, Y_COLUMNS,
    EMBEDDING_GLOVE, EMBEDDING_CRAWL, TRAIN_DATA, TEST_DATA,
    TRAIN_PREPROCESSED_DATA, TEST_PREPROCESSED_DATA)

from src.metrics import get_roc_auc
from src.utils import seed_torch, df_parallelize_run, save_data2json, cut_length, sigmoid, read_json
from src.weights import training_weights

from src.data_loader import load_embedding
from src.lstm_models.scheduler import CyclicLR
from src.lstm_models.models import LstmGruAtten
from src.preprocessing import preprocess_final
from src.tokenize import text2index

def main():

    # config settings

    config = read_json("../config/lstm_model_config.json")
    SEED = config.get('seed', 1029)
    seed_torch(seed=SEED)
    bn_sn_ratio = config.get('bn_sn_ratio', 4)
    maxlen = config.get('maxlen', 230)
    n_target = len(Y_COLUMNS)+1
    max_features = config.get('max_features', None)
    n_splits = config.get('n_splits', 10)
    n_epochs = config.get('n_epochs', 5)
    n_train_fold = config.get('n_train_fold', 5)
    batch_size = config.get('batch_size', 512)
    lstm_units = config.get('lstm_units', 128)
    step_size = config.get('step_size', 100)
    base_lr, max_lr = config.get('base_lr', 0.001), config.get('max_lr', 0.003)
    lr = config.get('lr', 0.001)
    enable_checkpoint_ensemble = config.get('enable_checkpoint_ensemble', True)
    save_checkpoint = config.get('save_checkpoint', True)

    LSTM_OUTPUT = OUTPUT_DIR / "LSTM_OUTPUT"
    LSTM_OUTPUT.mkdir(exist_ok=True)

    # preprocessing
    print("START PREPROCESSING")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    num_train_data = train_df.shape[0]

    txt_df = pd.concat([train_df[TEXT_COLUMN], test_df[TEXT_COLUMN]], axis=0)
    preprocessed_txt = df_parallelize_run(txt_df[TEXT_COLUMN].fillna(''), preprocess_final)
    train_df[TEXT_COLUMN] = preprocessed_txt[:num_train_data]
    test_df[TEXT_COLUMN] = preprocessed_txt[train_df.shape[0]:]

    train_df.to_csv(TRAIN_PREPROCESSED_DATA)
    test_df.to_csv(TEST_PREPROCESSED_DATA)

    vocab, text_sequences = text2index(preprocessed_txt, max_features)
    save_data2json(vocab, OUTPUT_DIR/'vocab.json')

    train_seq, test_seq = text_sequences[:num_train_data], text_sequences[num_train_data:]
    x_train = pad_sequences(train_seq, maxlen=maxlen, padding='pre', dtype='int32')
    x_test = pad_sequences(test_seq, maxlen=maxlen, padding='pre', dtype='int32')
    del train_seq, test_seq

    y_train = train_df[TOXICITY_COLUMN].values
    weights = training_weights(train_df, IDENTITY_COLUMNS, bn_sn_ratio)
    y_train = np.vstack([y_train, weights]).T.astype(np.float32)
    y_identity = train_df[IDENTITY_COLUMNS].fillna(0).values
    y_aux = train_df[[TOXICITY_COLUMN] + AUX_TOXICITY_COLUMNS].values
    y_train_total = np.hstack([y_train, y_aux, y_identity.values]).astype(np.float32)
    print("FINISH PREPROCESSING")

    # load embedding
    print("START EMBEDDING LOADING")

    glove_embedding = load_embedding(EMBEDDING_GLOVE, vocab, max_features)
    crawl_embedding = load_embedding(EMBEDDING_CRAWL, vocab, max_features)
    embedding_matrix = np.concatenate([glove_embedding, crawl_embedding], axis=1)
    np.savez_compressed(OUTPUT_DIR/"embedding_matrix.npz", embedding_matrix=embedding_matrix)
    del glove_embedding, crawl_embedding
    gc.collect()

    print("FINISH EMBEDDING LOADING")

    # training
    print("START TRAINING")

    loss_weight = 1.0/weights.mean()
    def custom_loss(y_pred, y_true):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss1 = nn.BCEWithLogitsLoss(weight=y_true[:, 1:2])(y_pred[:, 0:1], y_true[:, 0:1])
        bce_loss2 = nn.BCEWithLogitsLoss()(y_pred[:, 1:], y_true[:, 2:n_target])
        return (bce_loss1 * loss_weight) + bce_loss2

    loss_fn = custom_loss

    # matrix for the out-of-fold predictions
    train_preds = np.zeros((x_train.shape[0]))
    # matrix for the predictions on the test set
    test_preds = np.zeros((x_test.shape[0]))

    x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                  .split(x_train, y_train_total[:, 0] >= 0.5))

    test_preds_all = dict()
    # mask for sequence in a batch
    mask = torch.zeros((maxlen, 1), dtype=torch.long).cuda()

    for fold, (train_idx, valid_idx) in enumerate(splits):

        if fold == n_train_fold:
            break

        x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train_total[train_idx, :n_target], dtype=torch.float32).cuda()

        x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train_total[valid_idx, :n_target], dtype=torch.float32).cuda()

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        model = LstmGruAtten(embedding_matrix, y_aux.shape[-1], lstm_units, maxlen).cuda()

        param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=lr)

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=step_size,
                             mode='exp_range', gamma=0.99994)

        print(f'Fold {fold + 1}')
        valid_preds_fold_epoch = []
        test_preds_fold_epoch = []

        for epoch in range(n_epochs):

            start_time = time.time()

            model.train()
            avg_loss = 0.

            for i, (x_batch, y_batch) in enumerate(train_loader):

                # cut batch sequence padding
                x_batch = cut_length(x_batch, mask)

                y_pred = model(x_batch)

                scheduler.batch_step()

                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            elapsed_time = time.time() - start_time

            if save_checkpoint:
                checkpoint = {k: v for k, v in model.state_dict().items() if k != 'embedding.weight'}
                model_path = "Fold{}_epoch{}.pt".format(fold, epoch)
                torch.save(checkpoint, LSTM_OUTPUT/model_path)

            # eval
            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(valid_loader):
                x_batch = cut_length(x_batch, mask)
                with torch.no_grad():
                    y_pred = model(x_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            val_roc_score = get_roc_auc(y_train_total[valid_idx, 0],
                                        y_train_total[valid_idx, n_target:].astype('float32'), valid_preds_fold)


            print(
            '\nEpoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_final={:.4f} \t time={:.2f}s '
            '\n \t\t\t\t\t\t\t val_subgroup={:.4f}\n \t\t\t\t\t\t\t val_bpsn={:.4f} '
            '\n \t\t\t\t\t\t\t val_bnsp={:.4f} \n \t\t\t\t\t\t\t val_overall={:.4f} \t'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_roc_score['final_auc'], elapsed_time,
                val_roc_score['subgroup_auc'], val_roc_score['bpsn_auc'],
                val_roc_score['bnsp_auc'], val_roc_score['overall_auc'],
            ))
            valid_preds_fold_epoch.append(valid_preds_fold)

            # test
            test_preds_fold = np.zeros(len(x_test))
            for i, (x_batch,) in enumerate(test_loader):
                x_batch = cut_length(x_batch, mask)
                with torch.no_grad():
                    y_pred = model(x_batch).detach()

                test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            test_preds_fold_epoch.append(test_preds_fold)

        test_preds_all[fold] = test_preds_fold_epoch

        # checkpoint ensemble
        if enable_checkpoint_ensemble:
            checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
            valid_pred_ensemble = np.average(valid_preds_fold_epoch, axis=0, weights=checkpoint_weights)
            test_pred_ensemble = np.average(test_preds_fold_epoch, axis=0, weights=checkpoint_weights)
            val_roc_score = get_roc_auc(y_train_total[valid_idx, 0],
                                        y_train_total[valid_idx, n_target:].astype('float32'), valid_pred_ensemble)
            train_preds[valid_idx] = valid_pred_ensemble
            test_preds += test_pred_ensemble / n_train_fold
        else:
            val_roc_score = get_roc_auc(y_train_total[valid_idx, 0],
                                        y_train_total[valid_idx, n_target:].astype('float32'),
                                        valid_preds_fold_epoch[-1])
            train_preds[valid_idx] = valid_preds_fold_epoch[-1]
            test_preds += test_preds_fold_epoch[-1] / n_train_fold

        print(
            'Epoch {}/checkpoint_ensemble \t val_final={:.4f} \n '
            '\t\t\t\t val_subgroup={:.4f}\n \t\t\t\t val_bpsn={:.4f} '
            '\n \t\t\t\t val_bnsp={:.4f} \n \t\t\t\t val_overall={:.4f}'.format(
                epoch + 1, val_roc_score['final_auc'],
                val_roc_score['subgroup_auc'], val_roc_score['bpsn_auc'],
                val_roc_score['bnsp_auc'], val_roc_score['overall_auc'],
            ))

    print("FINISH TRAINING")

    # evaluate
    print("START EVALUATION")
    oof = train_df[[TOXICITY_COLUMN] + IDENTITY_COLUMNS].fillna(0) >= 0.5
    oof.loc[:, 'prediction'] = train_preds
    oof = oof.loc[oof.prediction != 0]
    final_auc = get_roc_auc(oof[TOXICITY_COLUMN].values, oof[IDENTITY_COLUMNS].values, oof.prediction.values)
    print("Cross validation result: {}".format(final_auc))

    # submission
    print("SUBMISSION GENERATION")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_preds
    })
    submission.to_csv(LSTM_OUTPUT/"lstm_submission.csv", index=False)
