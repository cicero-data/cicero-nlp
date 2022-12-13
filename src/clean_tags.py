'''
This file is used to clean the tags in the webpages stored in the target directory,
and save the cleaned webpages into a json file.
The extracted emails will also be saved into a json file and will be used for creating the synthetic email dataset.
'''
import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm


def get_parser(parser=argparse.ArgumentParser(
    description=
    "to clean the tags in the webpages stored in the target directory, and save the cleaned webpages into a json file."
),):
    parser.add_argument(
        "--input",
        type=str,
        default="webpages",
        help="the input directory to load the webpages",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./cleaned_texts",
        help="the output folder to save the cleaned texts and extracted emails",
    )
    return parser


def remove_tags_and_extract_emails(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    # extract emails
    pattern = re.compile("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    emails = re.findall(pattern, str(soup))

    for data in soup(['style', 'script']):
        # Remove style and script part
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings), emails


def clean_and_save(args):
    # detect if the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # create the dictionary for pure_texts
    pure_texts = collections.defaultdict()
    # create the dictionary for extracted_emails
    extracted_emails = collections.defaultdict()

    # iterating
    for file in tqdm(glob.glob(f"{args.input}/*.html")):
        politician_id = file.split("/")[-1].split(".")[0]
        with open(file, "r") as f:
            html = f.read()
            texts, emails = remove_tags_and_extract_emails(html)
            pure_texts[politician_id] = texts
            extracted_emails[politician_id] = emails

    # save the cleaned webpages
    with open(args.output / "pure_texts.json", "w") as f:
        json.dump(pure_texts, f)

    # save the cleaned webpages
    with open(args.output / 'extracted_emails.json', "w") as f:
        json.dump(extracted_emails, f)


if __name__ == "__main__":
    args = get_parser().parse_args()
    clean_and_save(args)
