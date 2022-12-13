'''
This file is used to BIO tagging on the pure texts for the mergeModelA, which covers information including names,
phone numbers and fax numbers.

The output file will be in the format of spaCy Doc file.

special tricks applied: 
1. use the regex to tag the phone numbers and fax numbers with multiple formats
   the spaCy ruler support tag with regex but the behavior is not exactly same as the vanilla regex.
   the vanilla regex is used to solve the problem.

special dependencies:
NONE

'''
import argparse
import collections
import json
import os
import re
from pathlib import Path

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans
from tqdm import tqdm


def get_parser(parser=argparse.ArgumentParser(
    description=
    "to BIO tagging on the pure texts for the mergeModelA, which covers information including names, phone numbers and fax numbers. The output file will be in the format of spaCy Doc file."
),):
    parser.add_argument(
        "--input_json",
        type=str,
        default="data.json",
        help="the input json file that stores the pure texts of the webpages",
    )
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
        "--ratio",
        type=float,
        default=0.1,
        help="the ratio of the dev set",
    )
    return parser


# helper function
def regex_search(web_content, unit, search_attribute, label, pattern_list,
                 error_log):
    """
    use regex to search the information in the web content and add the search result
    to the rule_pattern for bio tagging.

    Args:
        web_content: the content where we want to search information.
        unit: the dictionary contains all the attribute.
        search_attribute: the attribute's name. 
        label: the name of the label for the attribute.
        pattern_list: the pattern list which will be added to ruler.
        error_log:the dictionary to store the error instance.
    
    Returns:
        None
    """
    pattern_list.append({"label": label, 'pattern': unit[search_attribute]})
    part_list = re.split('[^\w]', unit[search_attribute])
    try:
        regex_pattern = r"((\(({})\)(\s){{0,1}}?)|(({})(-|.)))?({})(-|.)({})".format(part_list[1],\
                                                                          part_list[1],\
                                                                          part_list[3],\
                                                                          part_list[4])

        search_result = re.search(regex_pattern, web_content)
        if search_result:
            pattern_list.append({
                "label": 'PHONE',
                'pattern': search_result.group()
            })
    except Exception as e:

        politician_id = unit['id']

        error_log[politician_id]['url'] = unit['url_1']
        error_log[politician_id]['search_attribute'] = search_attribute
        error_log[politician_id]['error'] = str(e)


def bio_tag(args):
    # detect if the output directory exists
    if not args.output.exists():
        args.output.mkdir()

    # load the Cicero dataset
    cicero_df = pd.read_csv(args.input_csv, error_bad_lines=False)

    # load the pure texts of the webpages
    with open(args.input_json, "r") as f:
        pure_text_dict = json.load(f)

    nlp = spacy.load('en_core_web_sm')
    # bio tagging
    # the output file will be in the list first and then save as the spaCy Doc file
    tagged_data = []

    # create the error log
    error_log = collections.defaultdict(dict)

    for n in tqdm(range(len(cicero_df))):
        information_unit = cicero_df.iloc[n].dropna()
        politician_id = information_unit["id"]

        pure_text = pure_text_dict.get(str(politician_id), None)

        # skip the data point that we cannot find the pure text due to some reasons
        # for example, the url address is not valid
        if pure_text is None:
            continue

        ruler = nlp.add_pipe("span_ruler")
        bio_tag_pattern_list = []

        # reference: https://spacy.io/usage/rule-based-matching
        # tag the name
        # tag the full name, case insensitive by using the `LOWER` attribute
        bio_tag_pattern_list.append({
            "label":
                'NAME',
            'pattern': [{
                'LOWER': information_unit['first_name'].lower()
            }, {
                'LOWER': information_unit['last_name'].lower()
            }]
        })
        # tag the first name, case insensitive by using the `LOWER` attribute
        bio_tag_pattern_list.append({
            "label": 'NAME',
            'pattern': [{
                'LOWER': information_unit['first_name'].lower()
            }]
        })
        # tag the last name, case insensitive by using the `LOWER` attribute
        bio_tag_pattern_list.append({
            "label": 'NAME',
            'pattern': [{
                'LOWER': information_unit['last_name'].lower()
            }]
        })

        # tag the phone number
        if 'primary_phone_1' in information_unit.keys():
            regex_search(pure_text, information_unit, 'primary_phone_1',
                         'PHONE', bio_tag_pattern_list, error_log)

        if 'primary_phone_2' in information_unit.keys():
            regex_search(pure_text, information_unit, 'primary_phone_2',
                         'PHONE', bio_tag_pattern_list, error_log)

        if 'secondary_phone_1' in information_unit.keys():
            regex_search(pure_text, information_unit, 'secondary_phone_1',
                         'PHONE', bio_tag_pattern_list, error_log)

        if 'secondary_phone_2' in information_unit.keys():
            regex_search(pure_text, information_unit, 'secondary_phone_2',
                         'PHONE', bio_tag_pattern_list, error_log)

        # tag the fax number
        if 'primary_fax_1' in information_unit.keys():
            regex_search(pure_text, information_unit, 'primary_fax_1', 'FAX',
                         bio_tag_pattern_list, error_log)

        if 'primary_fax_2' in information_unit.keys():
            regex_search(pure_text, information_unit, 'primary_fax_2', 'FAX',
                         bio_tag_pattern_list, error_log)

        if 'secondary_fax_1' in information_unit.keys():
            regex_search(pure_text, information_unit, 'secondary_fax_1', 'FAX',
                         bio_tag_pattern_list, error_log)

        if 'secondary_fax_2' in information_unit.keys():
            regex_search(pure_text, information_unit, 'secondary_fax_2', 'FAX',
                         bio_tag_pattern_list, error_log)

        ruler.add_patterns(bio_tag_pattern_list)
        doc = nlp(pure_text)
        # split the doc into multiple chunks since the input of the model should be
        # less than 512 tokens
        length = len(doc) // 100
        for n in range(length + 1):
            sub_doc = nlp(str(doc[n * 100:(n + 1) * 100]))
            sub_doc.ents = filter_spans(sub_doc.spans["ruler"])
            tagged_data.append(sub_doc)

        # remove the ruler and initialize a new one for each politician/data point
        nlp.remove_pipe('span_ruler')

    # remove the empty data point
    tagged_data = [doc for doc in tagged_data if len(doc.ents) > 0]

    # split the tagged data into training set and test set
    train, dev = train_test_split(tagged_data,
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
    bio_tag(args)
