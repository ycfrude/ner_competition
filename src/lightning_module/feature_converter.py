from collections import Counter
from .config import Config
from transformers import AutoTokenizer
from typing import Union, List
import torch
import json

class FeatureConverter:
    def __init__(self, config :Config, tokenizer :AutoTokenizer, label2id, id2label=None):
        self.tokenizer = tokenizer
        self.label2id = label2id
        if id2label is None:
            id2label = {v:k for k,v in label2id.items()}
        self.id2label = id2label
        self.config = config
        self.markup = config.markup
        
    def convert_query_to_ids(self, query:List[Union[str, List[str]]]):
        max_length = self.config.max_length
        if isinstance(query[0], str):
            query = [query]
        input_lens = [min(len(x)+2,max_length) for x in query]
        query = [[xi.lower() if xi not in (" ", "") else "[SPACE]" for xi in x][:max_length] for x in query]
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in query]
        token_type_ids = [[0] * l for l in input_lens]
        attention_mask = [[1] * l for l in input_lens]
        inputs = {"input_ids":input_ids, "token_type_ids":token_type_ids, "attention_mask":attention_mask, "length":input_lens}
        return inputs
    
    def convert_label_to_ids(self, labels:List[Union[List[str], str]]):
        if isinstance(labels[0],str):
            labels = [labels]
        labels = [[self.label2id["O"]]+[self.label2id[xi] for xi in x] + [self.label2id["O"]] for x in labels]
        max_length = self.config.max_length
        labels = [x[:max_length] for x in labels]
        return labels
    
    def convert_ids_to_label(self, features:Union[List[Union[List[str], str]], torch.Tensor], length=None):
        if isinstance(features, torch.Tensor):
            features = features.numpy().tolist()
        if isinstance(features[0],int):
            features = [features]
        if length is not None and isinstance(length, list):
            assert len(length) == len(features), f"长度列表的长度为 {len(length)}, 与features的数量不一致 {len(features)}"
            features = [[self.id2label[xi] for xi in x][:length[i]] for i, x in enumerate(features)]
        elif isinstance(length, int):
            features = [[self.id2label[xi] for xi in x][:length] for i, x in enumerate(features)]
        else:
            features = [[self.id2label[xi] for xi in x] for i, x in enumerate(features)]
        return features
    
    def save_pretrained(self, pretrained_path):
        self.tokenizer.save_pretrained(pretrained_path)
        self.config.save_pretrained(pretrained_path)
        with open(pretrained_path+"/label2id.json", "w", encoding="utf8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=4)
        with open(pretrained_path+"/id2label.json", "w", encoding="utf8") as f:
            json.dump(self.id2label, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        config = Config.from_pretrained(pretrained_path)
        with open(pretrained_path+"/label2id.json", "r", encoding="utf8") as f:
            label2id = json.load(f)
            label2id = {k:int(v) for k,v in label2id.items()}
        with open(pretrained_path+"/id2label.json", "r", encoding="utf8") as f:
            id2label = json.load(f)
            id2label = {int(k):v for k,v in id2label.items()}
        return cls(config, tokenizer, label2id, id2label)
    
    @staticmethod
    def generate_label2id(labels=None):
        if labels is None:
            labels = ["X", 'O', 'B-1', 'I-1', 'B-2', 'I-2', 'B-3', 'I-3', 'B-4', 'I-4', 'B-5', 'I-5', 'B-6', 'I-6',
                'B-7', 'I-7', 'B-8', 'I-8', 'B-9', 'I-9', 'B-10', 'I-10', 'B-11', 'I-11', 'B-12', 'I-12', 'B-13',
                'I-13', 'B-14', 'I-14', 'B-15', 'I-15', 'B-16', 'I-16', 'B-17', 'I-17', 'B-18', 'I-18', 'B-19', 'I-19',
                'B-20', 'I-20', 'B-21', 'I-21', 'B-22', 'I-22', 'B-23', 'I-23', 'B-24', 'I-24', 'B-25', 'I-25', 'B-26',
                'I-26', 'B-28', 'I-28', 'B-29', 'I-29', 'B-30', 'I-30', 'B-31', 'I-31', 'B-32', 'I-32', 'B-33', 'I-33',
                'B-34', 'I-34', 'B-35', 'I-35', 'B-36', 'I-36', 'B-37', 'I-37', 'B-38', 'I-38', 'B-39', 'I-39', 'B-40',
                'I-40', 'B-41', 'I-41', 'B-42', 'I-42', 'B-43', 'I-43', 'B-44', 'I-44', 'B-46', 'I-46', 'B-47', 'I-47',
                'B-48', 'I-48', 'B-49', 'I-49', 'B-50', 'I-50', 'B-51', 'I-51', 'B-52', 'I-52', 'B-53', 'I-53', 'B-54',
                'I-54',"[START]", "[END]"]
        label2id = {}
        for i, label in enumerate(labels):
            label2id[label] = i 
        return label2id

    def get_entity_bios(self, seq):
        """Gets entities from sequence.
        note: BIOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        id2label = self.id2label
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks

    def get_entity_bio(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        id2label = self.id2label
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks

    def get_entities(self, seq):
        '''
        :param seq:
        :param id2label:
        :param markup:
        :return:
        '''
        markup = self.markup
        assert markup in ['bio','bios']
        if markup =='bio':
            return self.get_entity_bio(seq)
        else:
            return self.get_entity_bios(seq)

    def ner_score(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        markup = self.markup
        id2label = self.id2label
        origins, founds, rights = [], [], []
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = self.get_entities(label_path)
            pre_entities = self.get_entities(pre_path)
            origins.extend(label_entities)
            founds.extend(pre_entities)
            rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
        origin_counter = Counter([x[0] for x in origins])
        found_counter = Counter([x[0] for x in founds])
        right_counter = Counter([x[0] for x in rights])
        class_info = {}
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(origins)
        found = len(founds)
        right = len(rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return recall, precision, f1, class_info
    
    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1