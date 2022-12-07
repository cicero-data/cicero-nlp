'''
This file is used to BIO tagging on the address information. One customized dataset will be created.

The output file will be in the format of spaCy Doc file.

special tricks applied: 
1. insert the address together into the essays in the essay dataset
2. insert the entites from the CoNLL 2003 entities dataset into the essays in the essay dataset, to increase the difficulty of the task


special dependencies:
1. one essay dataset
2. CoNLL 2003 entities dataset

'''
import os
import spacy
from spacy.util import filter_spans
from spacy.tokens import Doc, DocBin
import collections
from tqdm import tqdm
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def get_parser(
    parser=argparse.ArgumentParser(
        description="to BIO tagging on the address information. One customized dataset will be created."
    ),
):
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data.csv",
        help="the input csv file that stores the Cicero dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="mergeA_data",
        help="the output directory that stores the tagged dataset",
    )
    parser.add_argument(
        "--essay",
        type=str,
        default="essay.tsv",
        help="the external file that stores the essay dataset",
    )
    parser.add_argument(
        "--conll",
        type=Path,
        default="conll",
        help="the external folder that stores the CoNLL 2003 dataset",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="the ratio of the dev set",
    )
    return parser


def insert_and_tag(args):
    # detect if the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load the Cicero dataset
    cicero_df = pd.read_csv(args.input_csv,  error_bad_lines=False)


    # load the essay dataset
    essay = pd.read_table(args.essay,  encoding='mac_roman')

    # load the CoNLL 2003 dataset
    doc_bin = DocBin().from_disk(args.conll/"train.spacy")
    nlp = spacy.load('en_core_web_sm')
    conll_docs = list(doc_bin.get_docs(nlp.vocab))
    conll_ents = [n.ents for n in conll_docs if len(n.ents)>0 ]

    # inserting and bio tagging
    # the output will be stored in the list first and then save as the spaCy Doc file
    inserted_data = []
    for n in tqdm(range(len(cicero_df))):
        attribute_list = list(cicero_df.iloc[n].dropna())
        attribute_dict = dict(cicero_df.iloc[n].dropna())

        # randomly select 6 entities from the CoNLL 2003 dataset
        random_ents = [list(n) for n in random.choices(conll_ents, k=6)]

        raw_essay = essay.iloc[n]['essay']

        address_list = []
        address1 = attribute_dict.get('primary_address_1')
        if address1: 
            address_list.append(address1)
        address2 = attribute_dict.get('primary_address_2')
        if address2: 
            address_list.append(address2)
        address3 = attribute_dict.get('primary_address_3')
        if address3: 
            address_list.append(address3)
        # reverse the address list since the primary address 3 is the beginning of the address    
        address_list = address_list[::-1]

        essay_sentences = [str(s) for s in nlp(raw_essay).sents]

        ruler = nlp.add_pipe("span_ruler")
        bio_tag_pattern_list = []

        chance = random.uniform(0,1)

        # 5% chance to insert the address scattedly into the essay    
        if chance<=0.05:
            random_ents = [str(n) for n in sum(random_ents, [])]
            attribute_list += random_ents

            # insert the address into the essay
            for m in range(len(attribute_list)):
                random_index = randrange(len(essay_sentences))
                essay_sentences = essay_sentences[:random_index] + [attribute_list[m]] + essay_sentences[random_index:]

            if 'primary_address_1' in attribute_dict:
                bio_tag_pattern_list.append({"label":'ADDRESS', 'pattern':attribute_dict['primary_address_1']})
            if 'primary_address_2' in attribute_dict:
                bio_tag_pattern_list.append({"label":'ADDRESS', 'pattern':attribute_dict['primary_address_2']})
            if 'primary_address_3' in attribute_dict:
                bio_tag_pattern_list.append({"label":'ADDRESS', 'pattern':attribute_dict['primary_address_3']})
        # 95% chance to insert the address together into the essay
        else:
            attribute_list = list(set(attribute_list) - set(address_list))
            random_ents = [str(n) for n in sum(random_ents, [])]
            #split the entities from conll into three arraies
            ent_array = np.array_split(random_ents,3)
            #convert the three arries into three strings
            long_ent = [' '.join(n) for n in ent_array]
            attribute_list = attribute_list + long_ent

            for m in range(len(attribute_list)):
                random_index = randrange(len(essay_sentences))
                essay_sentences = essay_sentences[:random_index] + [attribute_list[m]] + essay_sentences[random_index:]

            if address_list:
                address = ' '.join(address_list)
                random_index=randrange(len(essay_sentences))
                essay_sentences = essay_sentences[:random_index] + [address] + essay_sentences[random_index:]
                bio_tag_pattern_list.append({"label":'ADDRESS', 'pattern':address})

        ruler.add_patterns(bio_tag_pattern_list)
        essay_sentences = [str(n) for n in essay_sentences]
        text = ' '.join(essay_sentences)
        doc = nlp(text)
        doc.ents = filter_spans(doc.spans["ruler"])
        inserted_data.append(doc)

        # remove the ruler and initialize a new one for each politician/data point
        nlp.remove_pipe("span_ruler")
    
    # remove the empty data point
    interted_data = [doc for doc in interted_data if len(doc) > 0]

    # split the tagged data into training set and test set
    train, dev = train_test_split(tagged_data, test_size = args.ratio, random_state=42)

    train_db = DocBin()
    for n in train:
        train_db.add(n)
    train_db.to_disk(args.output / "train.spacy")

    dev_db = DocBin()
    for n in dev:
        dev_db.add(n)
    dev_db.to_disk(args.output / "dev.spacy")

if __name__ == "__main__":
    args = get_args().parse_args()
    insert_and_tag(args)