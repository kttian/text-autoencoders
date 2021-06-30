import argparse
import os 
import torch
import numpy as np
from utils import *
import nltk
import time

parser = argparse.ArgumentParser()
parser.add_argument('--outputs', metavar='FILE', required=True,
                    help='name of outputs file')
parser.add_argument('--gt', metavar='FILE', required=True,
                    help='name of ground truth file')

#######################################
### Part 1: Classification Accuracy ###
#######################################

present_tags = ["VBP", "VBZ", "MD"]
past_tags = ["VBD", "VBN"]
def classify_tagged(tagged):
    for word, tag in tagged:
        if tag in set(present_tags):
            return "present"
        elif tag in set(past_tags):
            return "past"
    return "none"

def classify_sent(sent):
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)
    return classify_tagged(tagged)

#############################################################
pres_path = "parallel_data/present_v2.txt"
past_path = "parallel_data/past_v2.txt"

# pres_path = "data/yelp/tense/valid.present"
# past_path = "data/yelp/tense/valid.past"

pres_sents = load_sent_no_split(pres_path)
past_sents = load_sent_no_split(past_path)

def eval(sents, gt):
    N = len(sents)
    num_tense = 0
    num_none = 0
    # counter = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        tense = classify_sent(sent)
        if tense == gt:
            num_tense += 1
        elif tense == "none":
            num_none += 1
        # elif counter < 5:
        #     print(tagged)
        #     counter +=1 
    return num_tense, num_none, N

def get_baselines():
    num_pres, num_none, N = eval(pres_sents, "present")
    print("present base accuracy:", num_pres/N)
    print("past:", N - num_pres - num_none)
    print("none:", num_none)
    print("total:", N)

    num_past, num_none, N = eval(past_sents, "past")
    print("past base accuracy:", num_past/N)
    print("present:", N - num_past - num_none)
    print("none:", num_none)
    print("total:", N)

# outputs_fn = "results_05_31_21/walk_arith_5_output.txt"
def get_accuracy(outputs_fn):
    outputs = load_sent_no_split(outputs_fn)
    num_past, num_none, N = eval(outputs, "past")
    print("output accuracy:", num_past/N)

#######################################
### Part 2: BLEU Score of Output ######
#######################################
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def get_bleu(outputs_pred, gt_fn):
    # outputs_pred = load_sent(outputs_fn)
    pasts_ref = load_sent(gt_fn)
    pasts_ref = [[x] for x in pasts_ref]
    bleu_score = corpus_bleu(pasts_ref, outputs_pred)
    print("bleu score:", bleu_score)

#######################################
### Part 3: Perplexity of Output ######
#######################################
# forward PPL = perplexity of LM trained on real data and evaluated on generated data
# reverse PPL = perplexity of LM trained on generated data and evaluated on real data
from test import calc_ppl
from batchify import get_batches2, get_batches, get_batches3
from vocab import Vocab
from steerability import load_model 

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

checkpoint_dir = "checkpoints/yelp/daae3/"
vocab_file = os.path.join(checkpoint_dir, 'vocab.txt')
#if not os.path.isfile(vocab_file):
#    Vocab.build(train_sents, vocab_file, args.vocab_size)

vocab = Vocab(vocab_file)
batch_size = 256

model = load_model(checkpoint_dir)

def calc_ppl(sents, m):
    batches, _ = get_batches(sents, vocab, batch_size, device)
    total_nll = 0
    i = 0
    with torch.no_grad():
        for inputs, targets in batches:
            # i+=1
            # print(i)
            total_nll += model.nll_is(inputs, targets, m).sum().item()
    n_words = sum(len(s) + 1 for s in sents)    # include <eos>
    return total_nll / len(sents), np.exp(total_nll / n_words)

def get_ppl(outputs_pred):
    ppl = calc_ppl(outputs_pred, 100)
    print("perplexity:", ppl)

import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)c
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)

def get_ppl(outputs_pred):
    # breakpoint()
    sample = 10
    scores = [score(x) for x in outputs_pred[:sample]]
    print("done w scores")
    base_pres = [score(x) for x in pres_sents[:sample]]
    print("done pres")
    base_past = [score(x) for x in past_sents[:sample]]
    print("done past")

    for i in range(5):
        print(outputs_pred[i])
        print(past_sents[i])
        print(i, ":", scores[i], base_pres[i], base_past[i])
    
    perp = sum(scores)/len(scores)
    perp_pres = sum(base_pres)/len(base_pres)
    perp_past = sum(base_past)/len(base_past)

    print("perplexity:", perp)
    print("--perp pres:", perp_pres)
    print("--perp past:", perp_past)

### main ###
def main(args):
    # start_time = time.perf_counter()
    # print(0)
    get_baselines()
    get_accuracy(args.outputs)
    # cur_time = time.perf_counter()
    # print(cur_time - start_time)

    outputs_pred = load_sent(args.outputs)
    get_bleu(outputs_pred, args.gt)

    # cur_time = time.perf_counter()
    # print(cur_time - start_time)
    # outputs_pred = load_sent_no_split(args.outputs)
    # print("hi")
    # get_ppl(outputs_pred)

    # cur_time = time.perf_counter()
    # print(cur_time - start_time)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
