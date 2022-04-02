import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, ElectraPreTrainedModel, ElectraModel

from .layers.crf import CRF
from .layers.lstm import LSTM


# from losses.focal_loss import FocalLoss
# from losses.label_smoothing import LabelSmoothingCrossEntropy

class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = "ce"
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            # if self.loss_type == 'lsr':
            #     loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss(ignore_index=0)
            # else:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
    
    def decode(self, inputs_ids=None, token_type_ids=None, attention_mask=None, logits=None):
        if logits is not None:
            logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        logits = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        return logits

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores
    
    def decode(self, inputs_ids=None, token_type_ids=None, attention_mask=None, logits=None):
        if logits is not None:
            return self.crf.decode(logits, attention_mask)
        else:
            logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            return self.crf.decode(logits, attention_mask)


class ElectraCrfForNer(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraCrfForNer, self).__init__(config)
        self.bert = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores
    
    def decode(self, inputs_ids=None, token_type_ids=None, attention_mask=None, logits=None):
        if logits is not None:
            return self.crf.decode(logits, attention_mask)
        else:
            logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            return self.crf.decode(logits, attention_mask)


class BertBiLstmCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiLstmCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = LSTM(input_size = config.hidden_size, hidden_size=config.hidden_size//2,
                         bidirectional=True, num_layers=2, batch_first=True,
                         dropout_prob=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output = self.lstm(sequence_output)
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores

    def decode(self, inputs_ids=None, token_type_ids=None, attention_mask=None, logits=None):
        if logits is not None:
            return self.crf.decode(logits, attention_mask)
        else:
            logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            return self.crf.decode(logits, attention_mask)
