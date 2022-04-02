# author: 
# contact: ycfrude@163.com
# datetime:2022/4/2 10:32 AM
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW, \
    AutoModel, AutoTokenizer, AutoModelForMaskedLM

from src.utils import ListDataset
from .data_collate import DataCollatorForLanguageModeling


class plMLMDataset(pl.LightningDataModule):
    def __init__(self, config, train_data, tokenizer):
        super(plMLMDataset, self).__init__()
        self.train_data = train_data
        self.config = config
        self.tokenizer = tokenizer
        self.collator = DataCollatorForLanguageModeling(tokenizer)

    def prepare_data(self):
        self.tokenizer.add_tokens("[unused1]", special_tokens=True)
        query = self.train_data
        max_length = self.config.max_length
        input_lens = [min(len(x) + 2, max_length) for x in query]
        inputs = [["[CLS]"]+[xi.lower() if xi not in (" ", "") else "[unused]" for xi in x][:input_lens[i]-2]+["[SEP]"] for i, x in enumerate(query)]
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in inputs]
        token_type_ids = [[0] * l for l in input_lens]
        attention_mask = [[1] * l for l in input_lens]
        special_tokens_mask = [[1] + [0] * (l-2) + [1] for l in input_lens]
        inputs = [{'input_ids':input_ids[i],
                   'token_type_ids':token_type_ids[i],
                   'attention_mask':attention_mask[i],
                   'special_tokens_mask':special_tokens_mask[i]}
                  for i in range(len(input_lens))]
        self.dataset = ListDataset(inputs)

    def train_dataloader(self):
        dataset = self.dataset
        batch_size=self.config.batch_size
        num_workers = self.config.get("num_workers", 4)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, collate_fn=self.collator)
        return dataloader


class plMLMModel(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.pretrained_path = config.pretrained_path
        self.model = model
        self._device = config.device
        self.markup = self.config.markup

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        input_lens = batch.pop("length", None)
        inputs = {k: v.to(self._device) for k, v in batch.items()}
        outputs = self(inputs)
        loss = outputs[0].mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        model = self.model
        config = self.config
        t_total = self.config.train_samples // self.config.batch_size * self.config.num_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        model_params = [self.model.named_parameters()]
        optimizer_grouped_parameters = []
        for i, params in enumerate(model_params):
            lr = config.learning_rate
            param_gorpups = [{'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                              'weight_decay': config.weight_decay, 'lr': lr},
                             {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                              'lr': lr}]
            optimizer_grouped_parameters += param_gorpups
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

    @classmethod
    def from_pretrained(cls, pretrained_path):
        from src.config import Config
        config = Config.from_pretrained(pretrained_path)
        model_type = config.model_type
        model = AutoModel.from_pretrained(pretrained_path)
        return cls(config, model)


def train(train_data, config, gpus=[1]):
    config.update(train_samples = len(train_data))
    pretrained_path = config.pretrained_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModelForMaskedLM.from_pretrained(pretrained_path)
    pl_dataset = plMLMDataset(config, train_data, tokenizer)
    pl_model = plMLMModel(config, model).to(config.device)
    checkpoint_callback = ModelCheckpoint(monitor='f1', mode='max')
    earlystopping_callback = EarlyStopping(monitor='f1', patience=1, check_on_train_epoch_end=False, mode="max")
    trainer = pl.Trainer(max_epochs=config.num_epochs, gpus=gpus, callbacks=[checkpoint_callback, earlystopping_callback],
                         gradient_clip_val=1, accumulate_grad_batches=config.accumulate_grad_batches, val_check_interval=0.5)
    trainer.fit(pl_model, pl_dataset)
    ckpt = torch.load(checkpoint_callback.best_model_path)
    pl_model.load_state_dict(ckpt["state_dict"])
    pl_model.save_pretrained(config.model_ckpt_path)
    pl_dataset.tokenizer.save_pretrained(config.model_ckpt_path)
