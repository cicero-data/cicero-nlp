'''
This file is used to clean the tags in the webpages stored in the target directory,
and save the cleaned webpages into a json file.
'''
import json
import collections
import os
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse

def get_parser(
    parser=argparse.ArgumentParser(
        description="to clean the tags in the webpages stored in the target directory, and save the cleaned webpages into a json file."
    ),
):
    parser.add_argument(
        "--input",
        type=str,
        default="webpages",
        help="the input directory to load the webpages",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_webpages.json",
        help="the output file name to save the cleaned webpages",
    )
    return parser

def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
  
    for data in soup(['style', 'script']):
        # Remove style and script part
        data.decompose()
  
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)
    

def clean_and_save(args):
    # create the cleaned webpages
    cleaned_webpages = collections.defaultdict()

    # clean the webpages
    for file in tqdm(glob.glob(f"{args.input}/*.html")):
        politician_id = file.split("/")[-1].split(".")[0]
        with open(file, "r") as f:
            html = f.read()
            cleaned_webpages[politician_id] = remove_tags(html)
    
    # save the cleaned webpages
    with open(args.output, "w") as f:
        json.dump(cleaned_webpages, f)

if __name__ == "__main__":
    args = get_parser().parse_args()
    clean_and_save(args)