from utils import *
import nltk
import numpy as np
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# nltk.help.upenn_tagset()


#######################################
### Part 1: Classification Accuracy ###
#######################################

# sentence = "the pizza is pretty bland, despite a hefty helping of oregano ."
# tokens = nltk.word_tokenize(sentence)
# print(tokens)

# tagged = nltk.pos_tag(tokens)
# print(tagged)

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

# tense = classify_tagged(tagged)
# print(tense)

#############################################################
pres_path = "parallel_data/present.txt"
past_path = "parallel_data/past.txt"

pres_sents = load_sent_no_split(pres_path)
past_sents = load_sent_no_split(past_path)

def eval(sents, gt):
    N = len(sents)
    num_tense = 0
    num_none = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        tense = classify_sent(sent)
        if tense == gt:
            num_tense += 1
        elif tense == "none":
            num_none += 1
    return num_tense, num_none, N 

num_pres, num_none, N = eval(pres_sents, "present")
print("present base accuracy:", num_pres/N)
# print("past:", N - num_pres - num_none)
# print("none:", num_none)
# print("total:", N)

num_past, num_none, N = eval(past_sents, "past")
print("past base accuracy:", num_past/N)


# N = len(pres_sents)
# npres = 0
# counter = 0
# print("---present----")
# for sent in pres_sents:
#     tokens = nltk.word_tokenize(sent)
#     tagged = nltk.pos_tag(tokens)
#     tense = classify_sent(sent)
#     if tense == "present":
#         npres += 1
#     elif counter < 5:
#         counter += 1
#         print(tagged)

# print("---past----")
# npres = 0
# counter = 0
# for sent in past_sents:
#     tokens = nltk.word_tokenize(sent)
#     tagged = nltk.pos_tag(tokens)
#     tense = classify_sent(sent)
#     if tense == "past":
#         npres += 1
#     elif counter < 5:
#         counter += 1
#         print(tagged)

# print("total:", N)
# print("correct:", npres)
# print("incorrect:", N - npres)
# print("accuracy:", npres/N)

outputs_fn = "results_05_31_21/walk_arith_5_output.txt"
outputs = load_sent_no_split(outputs_fn)

num_past, num_none, N = eval(outputs, "past")
print("output accuracy:", num_past/N)

#######################################
### Part 2: BLEU Score of Output ######
#######################################
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

outputs_pred = load_sent(outputs_fn)
pasts_ref = load_sent(past_path)
pasts_ref = [[x] for x in pasts_ref]
# print(outputs_pred[0:5])
# print(pasts_ref[0:5])
bleu_score = corpus_bleu(pasts_ref, outputs_pred)
print("bleu score:", bleu_score)

#######################################
### Part 3: Perplexity of Output ######
#######################################
# forward PPL = perplexity of LM trained on real data and evaluated on generated data
# reverse PPL = perplexity of LM trained on generated data and evaluated on real data
from test import calc_ppl
from batchify import get_batches2, get_batches, get_batches3
import os 
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

ppl = calc_ppl(outputs_pred, 100)
print("perplexity:", ppl)