import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from .models.bert_for_ner import BertCrfForNer, BertSoftmaxForNer, ElectraCrfForNer
from .feature_converter import FeatureConverter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .utils import collate_fn
import torch
import numpy as np

class DictDataset(Dataset):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
    
    def __len__(self):
        k = list(self.inputs.keys())[0]
        return len(self.inputs[k])
    
    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.inputs.items()}

class plNERDataset(pl.LightningDataModule):
    def __init__(self, config, train_data, val_data, feature_converter:FeatureConverter=None):
        super(plNERDataset, self).__init__()
        self.config = config
        self.feature_converter = feature_converter
        self.data = {"train":train_data, "val":val_data}
    
    def convert_query_to_ids(self, query:Union[str, List[str]]):
        return self.feature_converter.convert_query_to_ids(query)
    
    def convert_label_to_ids(self, labels:List[Union[List[str], str]]):
        return self.feature_converter.convert_label_to_ids(labels)
    
    def prepare_data(self):
        dataset = {}
        for mode, data in self.data.items():
            query = [x['query'] for x in data]
            label = [x['labels'] for x in data]
            inputs = self.convert_query_to_ids(query)
            label = self.convert_label_to_ids(label)
            inputs["labels"] = label
            dataset[mode] = DictDataset(inputs)
        self.dataset = dataset
        
    def train_dataloader(self):
        dataset = self.dataset["train"]
        batch_size=self.config.batch_size
        num_workers = self.config.get("num_workers", 4)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        return dataloader
    
    def val_dataloader(self):
        dataset = self.dataset["val"]
        batch_size=self.config.batch_size
        num_workers = self.config.get("num_workers", 4)
        dataloader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        return dataloader


class plNERModel(pl.LightningModule):
    def __init__(self, config, model, feature_converter:FeatureConverter):
        super().__init__()
        self.config = config
        self.pretrained_path = config.pretrained_path
        self.model = model
        self._device = config.device
        self.feature_converter = feature_converter
        self.markup = self.config.markup

    def forward(self, batch):
        return self.model(**batch)
    
    def decode(self, input_ids=None, token_type_ids=None, attention_mask=None, logits=None):
        return self.model.decode(input_ids, token_type_ids, attention_mask, logits)
    
    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        input_lens = batch.pop("length", None)
        inputs = {k: v.to(self._device) for k, v in batch.items()}
        outputs = self(inputs)
        loss = outputs[0].mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx=None, optimizer_idx=None):
        input_lens = batch.pop("length", None)
        inputs = {k: v.to(self._device) for k, v in batch.items()}
        outputs = self(inputs)
        loss, logits = outputs[:2]
        tags = self.decode(logits=logits, attention_mask=inputs['attention_mask'])
        labels = inputs['labels'].detach().cpu().numpy().tolist()
        input_lens = input_lens.cpu().numpy()
        tags = tags.squeeze(0).detach().cpu().numpy()
        # print("input_ids = ", inputs["input_ids"][0])
        # print("input_lens = ", input_lens[0])
        # print("tag = ", tags[0])
        # print("label = ", labels[0])
        # quit()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'labels':labels, 'tags':tags, 'input_lens':input_lens}
    
    def validation_epoch_end(self, outputs) -> None:
        labels = [x for output in outputs for x in output['labels']]
        tags = [x for output in outputs for x in output['tags']]
        input_lens = [x for output in outputs for x in output['input_lens']]
        losses = [output['loss'] for output in outputs]
        avg_loss = sum(losses) / len(losses)
        labels = [label[:input_lens[i]] for i, label in enumerate(labels)]
        tags = [tag[:input_lens[i]] for i, tag in enumerate(tags)]
        recall, precision, f1, class_info = self.feature_converter.ner_score(labels, tags)

        self.log('recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {"recall": recall, "precision":precision, "f1":f1, "class_info":class_info}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        model = self.model
        config = self.config
        t_total = self.config.train_samples // self.config.batch_size * self.config.num_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay, 'lr': config.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': config.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay, 'lr': config.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': config.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay, 'lr': config.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': config.crf_learning_rate}
        ]
        warmup_steps = int(t_total * config.warmup_rate)
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'  # called after each training step
        }
        return [optimizer], [scheduler]
    
    def save_pretrained(self, pretrained_path):
        self.model.save_pretrained(pretrained_path)
        self.config.save_pretrained(pretrained_path)
        self.feature_converter.save_pretrained(pretrained_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_path):
        from .config import Config
        config = Config.from_pretrained(pretrained_path)
        model_type = config.model_type
        model = train_model[model_type].from_pretrained(pretrained_path)
        feature_converter = FeatureConverter.from_pretrained(pretrained_path)
        return cls(config, model, feature_converter)
        
train_model = {"BertSoftmaxForNer":BertSoftmaxForNer,"BertCrfForNer":BertCrfForNer,"ElectraCrfForNer":ElectraCrfForNer}

def train(train_data, val_data, config, label2id, id2label, gpus=[1]):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    config.update(train_samples = len(train_data), num_labels=len(label2id))
    model_type = config.model_type
    pretrained_path = config.pretrained_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = train_model[model_type].from_pretrained(pretrained_path, num_labels=config.num_labels)
    feature_converter = FeatureConverter(config, tokenizer, label2id, id2label)
    pl_dataset = plNERDataset(config, train_data, val_data, feature_converter=feature_converter)
    pl_model = plNERModel(config, model, feature_converter=feature_converter).to(config.device)
    checkpoint_callback = ModelCheckpoint(monitor='f1', mode='max')
    earlystopping_callback = EarlyStopping(monitor='f1', patience=1, check_on_train_epoch_end=False, mode="max")
    trainer = pl.Trainer(max_epochs=config.num_epochs, gpus=gpus, callbacks=[checkpoint_callback, earlystopping_callback], gradient_clip_val=1, accumulate_grad_batches=4)
    trainer.fit(pl_model, pl_dataset)
    ckpt = torch.load(checkpoint_callback.best_model_path)
    pl_model.load_state_dict(ckpt["state_dict"])
    pl_model.save_pretrained(config.model_ckpt_path)

def predict(data, config, device=None):
    import os
    from tqdm import tqdm 
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    pl_model = plNERModel.from_pretrained(config.model_ckpt_path)
    feature_converter = FeatureConverter.from_pretrained(config.model_ckpt_path)
    query = [x['query'] for x in data]
    print(len(query))
    inputs = feature_converter.convert_query_to_ids(query)
    print(len(inputs))
    dataset = DictDataset(inputs)
    batch_size = config.batch_size
    num_workers = config.get("num_workers", 4)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
    device = config.device if device is None else device
    pl_model.eval()
    pl_model.to(device)
    all_tags = []
    for batch in tqdm(dataloader, desc="sentence tagging"):
        length = batch['length'].cpu().numpy().tolist()
        inputs = {k:v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        tags = pl_model.decode(**inputs).detach().cpu().squeeze(0).numpy().tolist()
        for i, tag in enumerate(tags):
            all_tags.append(tag[1:length[i]-1])
    all_tags = feature_converter.convert_ids_to_label(all_tags)
    print(len(all_tags))
    for i, example in enumerate(data):
        example["labels"] = all_tags[i]
    return data

