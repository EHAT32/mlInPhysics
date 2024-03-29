from LLmodel import LModel
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
nltk.download('punkt')

import numpy as np

def create_input_output_pairs(tokenized_text, sequence_length):
    pairs
    for i in range(len(tokenized_text) - sequence_length):
        input_sequence = tokenized_text[i:i+sequence_length]
        output_token = tokenized_text[i+sequence_length]
        pairs.append((input_sequence, output_token))
    return pairs


path = "C:/Users/roman/Downloads/updated_rappers.csv"

df = pd.read_csv(path)
lines_num = df.shape[0]

vocab_size = 30000
embedding_dim = 100
hidden_dim = 256
num_layers = 2

model = LModel(vocab_size=vocab_size, 
               embedding_dim=embedding_dim, 
               hidden_dim=hidden_dim, 
               num_layers=num_layers)

epoch_num = 10
iter_num = 1024

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
for epoch in range(epoch_num):
    for _ in range(iter_num):
        idx = np.random.randint(0, lines_num - 1)
        row = df.iloc[idx]
        line = row['lyric'] + ' ' + row['next lyric'] 
        line = word_tokenize(line)
        token_ids = tokenizer(line)
        if len(line) < 8:
            continue
        pairs = create_input_output_pairs(line, sequence_length=3)
