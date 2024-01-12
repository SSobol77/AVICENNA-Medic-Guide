import re
from typing import List
from collections import defaultdict

class Tokenizer:
    def __init__(self, special_tokens: List[str] = ["<pad>", "<unk>", "<s>", "</s>"]):
        self.special_tokens = special_tokens
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.word_count = defaultdict(int)

    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        # Improved tokenization using regular expressions
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def numericalize(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token2idx.get(token, self.token2idx["<unk>"]) for token in tokens]

    def add_vocabulary(self, texts: List[str]):
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_count[token] += 1

        frequent_tokens = [token for token, count in self.word_count.items() if count > 1]
        self.add_tokens(frequent_tokens)
