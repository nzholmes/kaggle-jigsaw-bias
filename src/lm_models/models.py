import numpy as np
import torch
from torch import nn

from ..pytorch_pretrained_bert import BertForSequenceClassification, BertModel

device = torch.device('cuda')


class BertForSequenceClassification_StaticWeight4(BertForSequenceClassification):

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config, num_labels)

        self.num_labels = num_labels
        self.bert = BertModel(config)
        encode_weights = np.array([0.8, 0.85, 0.9, 0.95], dtype=np.float32)
        weights = torch.from_numpy(encode_weights).to(device)
        self.weights = weights

        # pooling layer
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        # classifier layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        token_tensors = torch.stack([l[:, 0] * self.weights[i] for i, l in enumerate(encoded_layers[8:])], dim=1)
        # token_tensors = torch.mul(token_tensors, torch.unsqueeze(self.weights, -1))
        # token_tensors = torch.mean(token_tensors, dim=1)
        token_tensors = torch.sum(token_tensors, dim=1)
        pooled_output = self.dense(token_tensors)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class BertForSequenceClassification_LayerConcat4(BertForSequenceClassification):

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config, num_labels)

        self.num_labels = num_labels
        self.bert = BertModel(config)

        # pooling layer
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.activation = nn.Tanh()

        # classifier layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        encoded_layers = [l[:, 0] for i, l in enumerate(encoded_layers) if i >= 8]
        # mean_pool = torch.mean(encoded_layers, dim=1)
        token_tensors = torch.cat(encoded_layers, 1)

        pooled_output = self.dense(token_tensors)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertForSequenceClassification_SecondtoLast(BertForSequenceClassification):

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config, num_labels)

        self.num_labels = num_labels
        self.bert = BertModel(config)

        # pooling layer
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        # classifier layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        encoded_layers = [l[:, 0] for i, l in enumerate(encoded_layers) if i == 10]
        # mean_pool = torch.mean(encoded_layers, dim=1)
        token_tensors = encoded_layers[0]

        pooled_output = self.dense(token_tensors)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits