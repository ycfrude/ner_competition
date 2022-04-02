from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
    
    
def collate_fn(batch):
    all_lens = torch.LongTensor([example["length"] for example in batch])
    max_len = max(all_lens)
    all_attention_mask = torch.LongTensor([example["attention_mask"] + [0] * (max_len - len(example["attention_mask"])) for example in batch])
    all_input_ids = torch.LongTensor([example["input_ids"] + [0] * (max_len - len(example["input_ids"])) for example in batch])
    inputs = {"input_ids":all_input_ids, "attention_mask":all_attention_mask, "length":all_lens}
    if "token_type_ids" in batch[0]:
        all_token_type_ids = torch.LongTensor([example["token_type_ids"] + [0] * (max_len - len(example["token_type_ids"])) for example in batch])
        inputs["token_type_ids"] = all_token_type_ids
    if "labels" in batch[0]:
        all_labels = torch.LongTensor([example["labels"] + [0] * (max_len - len(example["labels"])) for example in batch])
        inputs["labels"] = all_labels
    return inputs