#from test import *
import argparse
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches


checkpoint_dir = "checkpoints/yelp/daae/"
parallel_data_dir = "parallel_data/"

vocab_file = os.path.join(checkpoint_dir, 'vocab.txt')
#if not os.path.isfile(vocab_file):
#    Vocab.build(train_sents, vocab_file, args.vocab_size)
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

print(model)

present_file = parallel_data_dir + "present.txt"
past_file = parallel_data_dir + "past.txt"
present_data = load_sent(present_file)
past_data = load_sent(past_file)
present_batches, _ = get_batches(present_data, vocab, batch_size, device)
past_batches, _ = get_batches(past_data, vocab, batch_size, device)
N = len(present_batches)

model.eval()

# hyper parameters 
alpha = 1
dim_emb = 128 
batch_size = 1
w = torch.rand(dim_emb)

num_epochs = 10
for e in range(num_epochs):

    total_loss = 0
    for i in range(N):
        x = present_batches[i][1]
        x_edit = present_batches[i][1]
        
        #print("x", x)
        #print("x_edit", x)
        mu, logvar, z, logits = model(x)
        new_latent = z + alpha * w
        logits, hidden = model.decode(new_latent, x)

        loss = model.loss_rec(logits, x_edit)
        total_loss += loss

print(total_loss)
