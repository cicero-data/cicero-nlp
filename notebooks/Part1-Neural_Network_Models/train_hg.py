'''
The file is used to train the model with huggingface framework.
'''
import os
from ast import literal_eval
from collections import defaultdict
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd

import argparse

import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import DatasetDict, Dataset, load_metric


import wandb

def get_parser(
    parser_class=argparse.ArgumentParser(
        description="Train a model on a NER",
    )
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="data directory")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                        help="model name")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size") 
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of epochs")    
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")       
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--device", type=str, default='cuda'if torch.cuda.is_available() else 'cpu',
                        help="device to use") 
    parser.add_argument("--save_dir", type=str, default=os.getcwd() + "/models",
                        help="directory to save model")
    parser.add_argument("--wandb", action="store_true", help='whether to use wandb')
    
    return parser

'''
TODOS:
1. figure out a way to save the label_list or label_dict to the dataset automatically.
eg. label_list = ['O', 'B-NAME', 'I-NAME', 'B-EMAIL', 'B-PHONE', 'I-PHONE', 'B-SALUTATION', 'I-SALUTATION']

'''
label_list = ['O', 'B-NAME', 'I-NAME', 'B-EMAIL', 'B-PHONE', 'I-PHONE', 'B-SALUTATION', 'I-SALUTATION']
label_dict = {label: i for i, label in enumerate(label_list)}

def convert_str_to_list(example):
    '''
    function to convert the str inside the pd dataframe to list
    '''

    example['words'] = literal_eval(example['words'])
    example['labels'] = literal_eval(example['labels'])
    return example

def load_data(args):
    '''
    function to load the data, convert the dataype of the columns to list, and split the data into train, val, test
    '''
    data_df = pd.read_csv(args.data_dir)
    data_df = data_df.apply(lambda x: (convert_str_to_list(x)), axis = 1) ## convert the str inside the pd dataframe to list
    dataset = Dataset.from_pandas(data_df)
    train_testvalid = dataset.shuffle(seed=args.seed).train_test_split(0.2)
    test_vaild = train_testvalid['test'].train_test_split(0.5)
    dataset_dict =DatasetDict({
                            'train': train_testvalid['train'], 
                            'valid': test_vaild['train'], 
                            'test': test_vaild['test']})

    return dataset_dict

def preprocess_data(args, dataset_dict, tokenizer):
    '''
    function to preprocess the data, tokenize the words and align the labels with the tokens
    '''
    tokenized_datasets = dataset_dict.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    return tokenized_datasets


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, 
                                                    is_split_into_words=True,)
    label_all_tokens = True
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_dict.get(label[word_idx]))
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_dict.get(label[word_idx]) if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main(args):
    ## tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    ## dataset
    dataset_dict = load_data(args)
    tokenized_datasets = preprocess_data(args, dataset_dict, tokenizer)

    ## save path
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"the model will be saved to {save_path}")
    
    ## wandb
    if args.wandb: wandb.init(project="vanilaNER", config = args, save_code = True)

    # training args
    model_name = args.model_name.split("/")[-1]
    train_args = TrainingArguments(
    run_name = f"{model_name}-finetuned-name_entity_recognition",
    output_dir = os.path.join(args.save_dir, f"{model_name}-finetuned-name_entity_recognition"),
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=args.lr,
    load_best_model_at_end=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    report_to="wandb" if args.wandb else None,
)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
    model,
    train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
    trainer.train()

    trainer.save_model(save_path)
    ## test
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
    if args.wandb: wandb.login({'test/precision': results["overall_precision"], 
                                'test/recall': results["overall_recall"], 
                                'test/f1': results["overall_f1"], 
                                'test/accuracy': results["overall_accuracy"]})

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)