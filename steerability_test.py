import argparse
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches2, get_batches, get_batches3

checkpoint_dir = "checkpoints/yelp/daae/"
parallel_data_dir = "parallel_data/"

vocab_file = os.path.join(checkpoint_dir, 'vocab.txt')
vocab = Vocab(vocab_file)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}['aae'](vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

model = get_model(checkpoint_dir + "model.pt")
model.eval()

present_file = parallel_data_dir + "test.present"
past_file = parallel_data_dir + "test.past"
present_data = load_sent(present_file)
past_data = load_sent(past_file)

data_batches, _ = get_batches2(present_data, past_data, vocab, 1, device)
word_batches, _ = get_batches3(present_data, past_data, vocab, batch_size, device)

def decode_logits(logits):
    words_inds = torch.argmax(logits, 2)
    words = [vocab.idx2word[ind] for ind in word_inds]
    return(words)

w = torch.load("walk.pt")
indices = list(range(len(data_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        x_present = data_batches[idx][0]
        print("test set:", word_batches[idx][0])
        x_past = data_batches[idx][1]
        mu, logvar, z, logits = model(x)
        new_latent = z + alpha * w
        logits, hidden = model.decode(new_latent, x)
        words = decode_logits(logits)
        print("outcome:", words)