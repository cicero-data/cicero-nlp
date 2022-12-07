'''
This file is used to BIO tagging on the email information. One reformatted dataset will be created.

The output file will be in the format of spaCy Doc file.

special tricks applied: 


special dependencies:
1. one dictionary file that stores the additional email information

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
        description="This file is used to BIO tagging on the email information. One reformatted dataset will be created."
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
        "--email",
        type=str,
        default="essay.tsv",
        help="the external file that stores the essay dataset",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="the ratio of the dev set",
    )
    return parser

#helpder function
def choice_excluding(lst, exception_index, taget_number):
    '''
    random choose a list of items from a list, excluding the item at the exception_index
    
    
    
    '''
    refined_list = lst[:exception_index] + lst[exception_index+1:]
    refined_list = [literal_eval(n) for n in refined_list if not (isinstance(n, float))]
    refined_list = [n for n in refined_list if n]
    return random.choices(refined_list, k = taget_number)

def random_and_tag(args):
    # detect if the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # load the Cicero dataset
    cicero_df = pd.read_csv(args.input_csv,  error_bad_lines=False)


    #random selecting and bio_tagging
    inserted_data = []
    for n in tqdm(range(len(cicero_df))):
        attribute_list = list(cicero_df.iloc[n].dropna())
        attribute_dict = dict(cicero_df.iloc[n].dropna())
        
        attribute_list = sum(choice_excluding(email_list, n, 3),[])[:5]
        if 'email_1' in attribute_dict:
            attribute_list += [attribute_dict.get('email_1')]
        if 'email_2' in attribute_dict:
            attribute_list += [attribute_dict.get('email_2')]
        random.shuffle(attribute_list)
        attribute_list = [attribute_dict.get('first_name')] + [attribute_dict.get('last_name')] + attribute_list
        ruler = nlp.add_pipe("span_ruler")
        patterns = []

        if 'email_1' in attribute_dict:
            patterns.append({"label":'EMAIL', 'pattern':attribute_dict['email_1']})
        if 'email_2' in attribute_dict:
            patterns.append({"label":'EMAIL', 'pattern':attribute_dict['email_2']})
        
        ruler.add_patterns(patterns)
        text = ' '.join(attribute_list)
        doc = nlp(text)
        doc.ents = filter_spans(doc.spans["ruler"])
        dataset = []
        dataset.append(doc)
        interted_data.append(doc)
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
    random_and_tag(args)