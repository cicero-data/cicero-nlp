'''
This file is used to BIO tagging on the email information. One synthetic dataset will be created.

The output file will be in the format of spaCy Doc file.

special tricks applied: 


special dependencies: <br>
1. The `extracted_emails.json` generated by `clean_tags.py`.

'''
import argparse
import collections
import json
import os
import random
from ast import literal_eval
from pathlib import Path
from random import randrange

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans
from tqdm import tqdm


def get_parser(parser=argparse.ArgumentParser(
    description=
    "This file is used to BIO tagging on the email information. One synthetic dataset will be created."
),):
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data.csv",
        help="the input csv file that stores the Cicero dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="email",
        help="the output directory that stores the tagged dataset",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="extracted_emails.json",
        help="the json file that stores the extracted emails",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="the ratio of the dev set",
    )
    return parser


#helper function
def add_email(example, email_dict):
    """
    add the email information to the example as a separate column.
    
    Args:
        example: the row in the cicero df.
        email_dict: the dictionary that stores the email information.
    
    Returns:
        the row with the email information added.
    
    """
    example['extracted_email'] = email_dict.get(str(example['id']))
    return example


#helpder function
def choice_excluding(lst, exception_index, taget_number):
    """
    random choose a list of items from a list, 
    excluding the item at the exception_index
    
    Args: 
      lst: lst for randomly choose emails from.
      exception_index: the index of the item that should be excluded.
      taget_number: the number of items that should be chosen.

    Returns:
        a list of emails from the lst.
    
    """
    refined_list = lst[:exception_index] + lst[exception_index + 1:]
    refined_list = [
        literal_eval(n) for n in refined_list if not (isinstance(n, float))
    ]
    refined_list = [n for n in refined_list if n]
    return random.choices(refined_list, k=taget_number)


def random_choice_and_tag(args):
    # detect if the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load the Cicero dataset
    cicero_df = pd.read_csv(args.input_csv, error_bad_lines=False)

    # load the extracted emails
    with open(args.email, 'r') as f:
        email_dict = json.load(f)

    # add the email information to the cicero_df
    cicero_df = cicero_df.apply(lambda x: add_email(x, email_dict), axis=1)

    # get the list of emails
    email_list = cicero_df['extracted_email'].tolist()
    email_list = [str(n) for n in email_list]

    nlp = spacy.load('en_core_web_sm')
    #random selecting and bio_tagging
    inserted_data = []
    for n in tqdm(range(len(cicero_df))):
        attribute_list = list(cicero_df.iloc[n].dropna())
        attribute_dict = dict(cicero_df.iloc[n].dropna())

        # radomly choose 3 list of emails for the email_list
        # and then cancatenate them together
        # the first 5 emails in the cancatenated list will be the email_list
        # these emails will be treated as NEGATIVE cases
        attribute_list = sum(choice_excluding(email_list, n, 3), [])[:5]

        # get the correct email from the cicero_df
        # these emails will be treated as POSITIVE cases
        if 'email_1' in attribute_dict:
            attribute_list += [attribute_dict.get('email_1')]
        if 'email_2' in attribute_dict:
            attribute_list += [attribute_dict.get('email_2')]

        # shuffle the list of emails
        random.shuffle(attribute_list)

        # add the name of the politician at the beginning of the list
        # since the position of name is fixed, the language model will learn it as the hint
        # to identify the email
        attribute_list = [attribute_dict.get('first_name')
                         ] + [attribute_dict.get('last_name')] + attribute_list
        ruler = nlp.add_pipe("span_ruler")
        bio_tag_pattern_list = []

        if 'email_1' in attribute_dict:
            bio_tag_pattern_list.append({
                "label": 'EMAIL',
                'pattern': attribute_dict['email_1']
            })
        if 'email_2' in attribute_dict:
            bio_tag_pattern_list.append({
                "label": 'EMAIL',
                'pattern': attribute_dict['email_2']
            })

        ruler.add_patterns(bio_tag_pattern_list)
        input_text = ' '.join(attribute_list)
        doc = nlp(input_text)
        doc.ents = filter_spans(doc.spans["ruler"])
        dataset = []
        dataset.append(doc)
        inserted_data.append(doc)
        # remove the ruler and initialize a new one for each politician/data point
        nlp.remove_pipe("span_ruler")

    # remove the empty data point
    inserted_data = [doc for doc in inserted_data if len(doc) > 0]

    # split the tagged data into training set and test set
    train, dev = train_test_split(inserted_data,
                                  test_size=args.ratio,
                                  random_state=42)

    train_db = DocBin()
    for n in train:
        train_db.add(n)
    train_db.to_disk(args.output / "train.spacy")

    dev_db = DocBin()
    for n in dev:
        dev_db.add(n)
    dev_db.to_disk(args.output / "dev.spacy")


if __name__ == "__main__":
    args = get_parser().parse_args()
    random_choice_and_tag(args)
