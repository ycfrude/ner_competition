import json
import os

import torch


class Config:
    def __init__(self, params: dict = None, **kwargs) -> None:
        self.num_epochs = 10
        self.warmup_rate = 0.05
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
        self.val_batch_size = 128
        self.learning_rate = 4e-5
        self.crf_learning_rate = 1e-3
        self.eps = 1e-10
        self.max_length = 128
        self.model_type = "BertCrfForNer"
        self.pretrained_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/checkpoints/"
        self.model_ckpt_path = "checkpoints/outputs/ner_bert/"
        self.markup = 'bio'
        self.weight_decay = 0.01
        self.num_workers = 4
        self.accumulate_grad_batches = 4
        self.loss_type = "ce"
        self.config = dict(params) if params is not None else dict()
        
        self.__getattributes()
        self.__setattributes(params)
        if kwargs:
            self.update(**kwargs)

    def update(self, params: dict = None, **kwargs):
        if params is not None:
            self.config.update(params)
        if kwargs is not None:
            self.config.update(**kwargs)
        self.__setattributes(params, **kwargs)

    def __setattributes(self, params: dict = None, **kwargs):
        if params is not None:
            for k, v in params.items():
                setattr(self, k, v)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def __getattributes(self):
        for k,v in self.__dict__.items():
            if type(v) in (int, str, list, tuple, float):
                self.config[k] = v
    
    def get(self, key, value=None):
        if getattr(self, key):
            return self.__dict__[key]
        else:
            return value

    def save_pretrained(self, pretrained_path):
        mkdir(pretrained_path)
        with open(pretrained_path + "/args.json", 'w', encoding="utf8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        with open(pretrained_path + "/args.json", "r") as f:
            config = json.load(f)
        config = {k: type_transform(v) for k, v in config.items()}
        return cls(config)

    def __getitem__(self, idx):
        return self.__dict__[idx]
    
    
def mkdir(path: str):
    path_list = path.split('/')
    half_path = path_list[0]
    if half_path != '' and not os.path.exists(half_path):
        os.mkdir(half_path)
    for tmp_path in path_list[1:]:
        half_path = half_path + '/' + tmp_path
        if not os.path.exists(half_path):
            os.mkdir(half_path)
            
def type_transform(var):
    try:
        return int(var)
    except:
        try:
            return float(var)
        except:
            return str(var)