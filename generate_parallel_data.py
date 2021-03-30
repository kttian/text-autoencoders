import argparse
from tenseflow import change_tense
from utils import set_seed, logging, load_sent, load_sent_no_split, write_sent_no_split

parser = argparse.ArgumentParser()

''' parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')

parser.add_argument('--data', metavar='FILE', required=True,
                    help='path to data folder') '''

tense_data_dir = "data/yelp/tense/"
parallel_data_dir = "parallel_data/"

def convert(past_flag):
    tenses = ["present", "past"]
    tense_idx = int(past_flag)
    
    tense_data = tense_data_dir + "valid." + tenses[tense_idx]
    tense_sents = load_sent_no_split(tense_data)
    converted_sents = []
    for sent in tense_sents:
        changed = change_tense(sent, 1-tense_idx)
        converted_sents.append(changed)
    return converted_sents

def main(args):
    tense_data_dir = "data/yelp/tense/"
    parallel_data_dir = "parallel_data/"
    past_sents = load_sent_no_split(tense_data_dir + "valid.past")
    present_sents = []
    for sent in past_sents:
        changed = change_tense(sent, 'present')
        present_sents.append(changed)

    more_present = load_sent_no_split(tense_data_dir + "valid.present")
    present_sents += more_present
    for sent in more_present:
        changed = change_tense(sent, 'past')
        past_sents.append(changed)

    write_sent_no_split(past_sents, parallel_data_dir + "past.txt")
    write_sent_no_split(present_sents, parallel_data_dir + "present.txt")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)