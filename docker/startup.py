import sys
import os
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification


def main(argv):
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-large-NER')
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-large-NER')
    path = '/train/model/'
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print('path created.')
    tokenizer.save_pretrained(path)
    model.bert.save_pretrained(path)


if __name__ == "__main__":
    main(sys.argv)