"""
This code is used to scrape the webpages by using the url address stored in the Cicero dataset.
The webpages will be stored as HTML files in the taget directory.
The error log will be stored in the <target_directionary>/error_log.json file 
saying the (1) politician ID, (2) url address, (3) error message.
"""
import json
import collections
from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse

def get_parser(
    parser=argparse.ArgumentParser(
        description="to scrape the webpages by using the url address stored in the Cicero dataset."
    ),
):
    parser.add_argument(
        "--input",
        type=str,
        default="cicero_dataset.csv",
        help="the input file name of the Cicero dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="webpages",
        help="the target directory to store the webpages",
    )
    return parser

def scrape_and_save(args):
    # detect if the target directory exists
    if not os.path.exists(args.target):
        os.makedirs(args.target)

    # load the Cicero dataset
    df = pd.read_csv(args.input)

    # create the error log
    error_log = collections.defaultdict(dict)

    # scrape the webpages
    for i in tqdm(range(len(df))):
        politician_id = df.iloc[i]["id"]
        url = df.iloc[i]["url_1"]
        name = df.iloc[i]["name"] 
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            with open(f"{args.target}/{politician_id}.html", "w", encoding='utf-8') as f:
                f.write(str(soup))
        except Exception as e:
            error_log[politician_id]["url"] = url
            error_log[politician_id]["name"] = name
            error_log[politician_id]["error"] = str(e)

    # save the error log
    with open(f"{args.target}/error_log.json", "w") as f:
        json.dump(error_log, f)

if __name__ == "__main__":
    args = get_parser().parse_args()
    scrape_and_save(args)