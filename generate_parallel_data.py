import argparse
from tenseflow import change_tense
from utils import set_seed, logging, load_sent, load_sent_no_split, write_sent_no_split

parser = argparse.ArgumentParser()

tense_data_dir = "data/yelp/tense/"
parallel_data_dir = "parallel_data/"

def convert(tense_idx):
    tenses = ["present", "past"]
    tense_data = tense_data_dir + "valid." + tenses[tense_idx]
    tense_sents = load_sent_no_split(tense_data)
    converted_sents = []
    for sent in tense_sents:
        changed = change_tense(sent, tenses[1-tense_idx])
        converted_sents.append(changed)
    return tense_sents, converted_sents

def main(args):
    past_sents, present_sents = convert(1) # converts past to present
    present2, past2 = convert(0) # converts present to past
    past_sents += past2
    present_sents += present2 
    
    write_sent_no_split(past_sents, parallel_data_dir + "past.txt")
    write_sent_no_split(present_sents, parallel_data_dir + "present.txt")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)