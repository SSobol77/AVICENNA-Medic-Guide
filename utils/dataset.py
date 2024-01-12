import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

# It is assumed that the Tokenizer is defined in tokenization.py
from .tokenization import Tokenizer

class CustomDataset(Dataset):
    def __init__(self, src_texts: List[str], trg_texts: List[str], tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_inputs = [self.tokenizer.numericalize("<s> " + text + " </s>") for text in src_texts]
        self.trg_inputs = [self.tokenizer.numericalize("<s> " + text) for text in trg_texts]
        self.trg_outputs = [self.tokenizer.numericalize(text + " </s>") for text in trg_texts]

    def __len__(self):
        return len(self.src_inputs)

    def __getitem__(self, idx):
        src = self.src_inputs[idx]
        trg_input = self.trg_inputs[idx]
        trg_output = self.trg_outputs[idx]
        src = self.pad_sequence(src)
        trg_input = self.pad_sequence(trg_input)
        trg_output = self.pad_sequence(trg_output)
        return {"src": src, "trg_input": trg_input, "trg_output": trg_output}

    def pad_sequence(self, sequence):
        sequence = sequence[:self.max_length - 1]
        sequence += [self.tokenizer.token2idx["<pad>"]] * (self.max_length - len(sequence))
        return sequence

def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# Example usage:
# tokenizer = Tokenizer()
# dataset = CustomDataset(["Hi, my friend!"], ["Hello, how are you?"], tokenizer, max_length=128)
# data_loader = create_data_loader(dataset, batch_size=32)
