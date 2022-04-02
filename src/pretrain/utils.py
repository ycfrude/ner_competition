# author: 
# contact: ycfrude@163.com
# datetime:2022/4/2 10:39 AM
from typing import Union, List


class MLMTokenizer:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

    def tokenize(self, query:Union[List[str], str]):
        max_length = self.config.max_length
        input_lens = [min(len(x) + 2, max_length) for x in query]
        query = [["[CLS]"] + [xi.lower() if xi not in (" ", "") else "[SPACE]" for xi in x][:input_lens[i] - 2] + [
            "[SEP]"] for i, x in enumerate(query)]
        labels = [self.tokenizer.convert_tokens_to_ids(x) for x in query]
        token_type_ids = [[0] * l for l in input_lens]
        attention_mask = [[1] * l for l in input_lens]
        special_tokens_ids = [[1] + [0] * (l-2) + [1] for l in input_lens]
        return {"input_ids":labels,
                "token_type_ids":token_type_ids,
                "attention_mask":attention_mask,
                "special_tokens_ids":special_tokens_ids}

    def mlm_collate_fn(self, batch):
        max_length = self.config.max_length
        mlm_rate = self.config.get("mlm")
        vocab_size = len(self.tokenizer)
        all_lens = torch.LongTensor([example["length"] for example in batch])
        max_len = max(all_lens)
        all_attention_mask = torch.LongTensor(
            [example["attention_mask"] + [0] * (max_len - len(example["attention_mask"])) for example in batch])
        all_labels = torch.LongTensor(
            [example["input_ids"] + [0] * (max_len - len(example["input_ids"])) for example in batch])
        all_labels = torch.LongTensor(
            [example["input_ids"] + [0] * (max_len - len(example["input_ids"])) for example in batch])

        all_special_tokens_ids = torch.LongTensor(
            [example["special_tokens_ids"] + [0] * (max_len - len(example["special_tokens_ids"])) for example in batch])
        inputs = {"input_ids":all_input_ids, "attention_mask":all_attention_mask, "length":all_lens}
        if "token_type_ids" in batch[0]:
            all_token_type_ids = torch.LongTensor([example["token_type_ids"] + [0] * (max_len - len(example["token_type_ids"])) for example in batch])
            inputs["token_type_ids"] = all_token_type_ids
        if "labels" in batch[0]:
            all_labels = torch.LongTensor([example["labels"] + [0] * (max_len - len(example["labels"])) for example in batch])
            inputs["labels"] = all_labels
        return inputs