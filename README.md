# [30th place solution to Kaggle Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/leaderboard)

## Outline of the final solution

Final ensemble - average ensemble of 15 models:

* 6 LSTM-based models
* 9 BERT models (base only models)

More details: 

* LSTMs were trained on glove and facebook crawl embeddings.
* LSTM models were trained on different folds and for each fold, the checkpoint of model at each epoch were saved for checkpoint ensemble.
* Only Bert base uncased were fintuned and the varieties came from the outputs from different layers.
* Did [language model finetuning for Bert](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning) on competition text, which didn't give much help.
* GPT2 models were fine-tuned but were not in final model ensemble, which should have been included.

Some of the ideas came from discussions and kernels in the competition forum and the paper: [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf)


## The loss

All of the models were trained with a combination of weighted BCE loss and auxiliary loss. 

```
 def custom_loss(y_pred, y_true):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss1 = nn.BCEWithLogitsLoss(weight=y_true[:, 1:2])(y_pred[:, 0:1], y_true[:, 0:1])
    bce_loss2 = nn.BCEWithLogitsLoss()(y_pred[:, 1:], y_true[:, 2:n_target])
    return (bce_loss1 * loss_weight) + bce_loss2
```

The auxiliary data were `'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'`.

Based on two criteria: whether instances were in `subgroup: male, female, homosexual_gay_or_lesbian, christian, jewish, black, white, psychiatric_or_mental_illness` and whether their labels were `positive`, instances were categorized into four types  : `subgroup negative`, `subgroup positive`, `background negative`, `background positive`.

I found models did not perform well on negative instances in subgroup, i.e `subgroup_negative`. So I tuned instance weights so that the ratio of `subgroup_negative` and `background_negative` is 3, 4 and 5, making models pay more attention to such subgroup negative instances. This also increased model varieties in final ensemble. 

```
weights = np.ones((train_df.shape[0],), dtype=np.float32) * weight_coeff['all']

# Subgroup
weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) * \
           weight_coeff['sub']

# Subgroup Negative
weights += sn_identity.mean(axis=1).astype(bool).astype(int) * weight_coeff['sn']

# Background Positive
weights += bp_identity.all(axis=1).astype(int) * weight_coeff['bp']
```

Another aspect worth a note is that soft labels did better than hard labels. This was similar to [label smoothing](https://leimao.github.io/blog/Label-Smoothing/), which reduced overfitting and helped generalization.

## Preprocessing

Preprocessing was done in a way to increase the coverage of tokens in the embedding vocabulary.

Some helpful preprocessing:

* Clean contractions like `we'll`
* Clean misspelled words like `culturr -> culture`
* Clean special punctuations like `\uf0a7`
* Clean bad-case words
* Replace number not in embedding with `NUM_##`
* Replace web url not in embedding with `URL_##`

## LSTM-based models

### Architecture

The LSTM model architecture consisted of 2 bidirectional LSTM layers and a dense layer on top of that. An extra attention layer was applied on top of the second LSTM layer. The max pooling, average pooling from the second LSTM layer and the attention output were concatenated to form the input into the last dense layer. 

### Embeddings

The following embeddings were used:

* glove.840B.300d (glove vectors)
* crawl-300d-2M (fasttext)

The important trick here was to look up various word forms in embedding vocabulary to increase the coverage of tokens in the embedding vocabulary.

```
ps = PorterStemmer()
ls = LancasterStemmer()
sb = SnowballStemmer("english")

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
```

### Training details:

* Scheduler: Cyclic learning rate
* Dynamic bucketing
* Maxlen: 230
* 10 fold training

## BERTs

### Tricks to accelerate fine-tuning and inference

In fine-tuning, [mixed precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) and [gradient accumulation](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) were used with the help of Nvidia `apex`. Besides that, `dynamic bucketing` was also really helpful in reducing fine-tuning time. Basically, what dynamic bucketing does is not to pad all sequences to the same `max_len` but to pad sequences in the same batch to the larget length in this batch. Another variety of `dynamic bucketing` I saw in this competition was to order all training sequences by lengt, cut them into batches and shuffle different batches, which was able to both to improve speed and avoid overfitting.

I only fine-tuned `bert-base-uncased` model due to the limit of computing power. The model varieties came from differnt layer outputs. I found the output from the second to last layer achieved better result than that from the last layer. I also took concataneted outputs and weighted outputs from the last four layers. Combined with varieties of instance weights, this helped final model ensemble.

In inference, to squeeze as many models into ensemble as possible, I sorted sequences by length so that shorter ones went through model first, which shortened the inference time by 5 times.

```
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
```

### Training details

* Epoch: 2
* Batch size: 64
* Gradient accumulation step: 2
* Dynamic bucketing
* Optimizer: BertAdam, FusedAdam
* lr: 2e-5
* Scheduler: Warmup linear

## What I learned

* Pytorch Bert language model fine-tuning on training data
* Pytorch Bert fine-tuning and prediction
* Weighted BCEloss + aux_loss in pytorch
* [Dynamic bucketing and trimming in prediction](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779#latest-553355)
* [Mixed precision with Apex](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) 
* [Gradient accumulation](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) 


## What can be improved

* More varieties of embeddings:
	- Trained embeddings on training data
	- Character level embedding
* More varieties of model ensemble
	- Bert (base, large, uncased, cased, WWM whole word masking, fine-tune)
	- GPT2(cnn head, linear head)
	- XLNET
	- Different seeds
* More varieties of text
	- Different sentence length
	- Head+tail tokens
* [Exponential moving average for LSTM model parameters](https://github.com/sakami0000/kaggle_jigsaw/blob/master/src/lstm_models/utils.py#L201). For more information, read the [kaggle notebook](https://www.kaggle.com/sakami/single-lstm-pseudo-label).
* Leave out-of-fold data to find optimal parameters weights on validation data using optuna or some other packages. [Example from 3rd solution](https://github.com/sakami0000/kaggle_jigsaw/blob/master/compute_blending_weights.py)
* Dynamic learning rate decay for Bert layers like the solution of [3rd place](https://github.com/sakami0000/kaggle_jigsaw/blob/master/src/language_models/utils.py#L9)
* Lower learning rate in the 2nd epoch of Bert fine-tuning
* [Power geometric ensemble on ranking metrics such as AUC](https://medium.com/data-design/reaching-the-depths-of-power-geometric-ensembling-when-targeting-the-auc-metric-2f356ea3250e)
	> Visualizing the correlation between your models that will be used to ensemble is essential you understand the risk you are taking when ensembling using power average. The higher the decorrelation, the higher the risk of messing up things. Power average works better for highly correlated submissions. Power average may have worked for ranking performance metrics such as normalized Gini as well as AUC. Power averaging only for AUC, nothing else (I might have missed some exotic performance metrics that may be optimized through power averaging though). In AUC you need optimize to get the most positives at the top of the rank (i.e you need the lowest amount of false positives at the highest possible probabilities). When dealing with power averaging, you must keep in mind you are looking at how close your models are from 0, and not how close they are from 1 (due to the power function properties). It also means that a bad model used for ensembling can jinx your whole ensemble very harshly, even if you have 10 strong models that are supposed to “average out the error”.
* [Add `toxicity_annotator_count.min()/toxicity_annator_count` in aux_loss](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100530), which improved performance by ~0.055%
* [Take `toxicity_annotator_count` and `identity_annotator_count` into account and set higher weights for those with higher numbers on these two](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100530)

## Tricks in other top solutions

* How to set sample weights:
	- Based on `toxicity_annotator_count`. See solutions of 3rd and 11th place
	- Based on validation prediction error on different identities. See solutons of 11th and 16th place
* How to set loss:
	- Use soft numeric target instead of hard binary target
	- Bin target toxic score into multiple classes as targets and have weighted sum of prediction as final prediction. See solutions of 2nd place (multi-class prediction) and 8th place (multi-label prediction). I tried these two varieties but found multi-class predcition hurt and multi-label prediction gave almost the same performance.
* Knowledge distillation. See 33th place solution.
* Stacking lightgbm etc. See solutions of 9th, 14th and 27th place.
* Power geometric ensemble. See solutions of 2nd and 26th place.
* Bias mitigation (Include non-toxic text mentioning identity from wikipedia). See 4th solutions.
* [Multi-sample dropout](https://arxiv.org/abs/1905.09788) to increase robustness in 8th place.


