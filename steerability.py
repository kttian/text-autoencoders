#from test import *
import argparse
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches2, get_batches, get_batches3
import time

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

model.train()

# hyper parameters 
alpha = 1
dim_emb = 128 
batch_size = 32 # 256
#breakpoint()
present_batches, _ = get_batches(present_data, vocab, batch_size, device)
past_batches, _ = get_batches(past_data, vocab, batch_size, device)
data_batches, _ = get_batches2(present_data, past_data, vocab, batch_size, device)
word_batches, _ = get_batches3(present_data, past_data, vocab, batch_size, device)

B = len(data_batches)
#breakpoint()
print("batches", B)
w = torch.rand(dim_emb, requires_grad=True, device=device)
opt = optim.SGD([w], lr=0.1, momentum=0.9)

num_epochs = 20
start_time = time.perf_counter()
for e in range(num_epochs):
    total_loss = 0
    indices = list(range(len(data_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        #print(i, idx)
        x = data_batches[idx][0]
        x_edit = data_batches[idx][1]
        #print("x", x.shape)
        #print("x_edit", x_edit.shape)
        #print(word_batches[idx][0][0])
        #print(word_batches[idx][1][0])

        #print("x", x)
        #print("x_edit", x)
        mu, logvar, z, logits = model(x)
        #print("logits1", logits.shape)
        #print("z", z.shape)
        new_latent = z + alpha * w
        logits, hidden = model.decode(new_latent, x)
        #print("logits", logits.shape, logits.type())
        #print("x edit", x_edit.shape, x_edit.type())
        #print(x_edit)
        #breakpoint()
        #losses = model.autoenc(logits, x_edit)
        loss = model.loss_rec(logits, x_edit).mean()
        #print("loss", loss.shape, loss)
        #print("total loss", total_loss)
        #print("walk", w)
        #print("--------")

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss

    print("---------------------------")
    print("FINISHED EPOCH", e)
    print("loss", total_loss/B)
    epoch_time = time.perf_counter()
    print("time", epoch_time)

print("FINISHED TRAINING")
print(w)
torch.save(w, "walk_lr01.pt")
print(total_loss)
