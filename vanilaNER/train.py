import os
from ast import literal_eval
from collections import defaultdict
from itertools import count
from pathlib import Path

import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    parser.add_argument("--max_len", type=int, default=128,
                        help="max length of input sequence")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size") 
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of epochs")    
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")       
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--device", type=str, default='cuda'if torch.cuda.is_available() else 'cpu',
                        help="device to use") 
    parser.add_argument("--save_dir", type=str, default=os.getcwd() + "./models",
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


## helper functions which are used specifically for this task, training vanilla NER models
def convert_str_to_list(example):
    '''
    function to convert the str inside the pd dataframe to list
    '''

    example['words'] = literal_eval(example['words'])
    example['labels'] = literal_eval(example['labels'])
    return example

def tokenize_and_align_labels(examples, tokenizer):
    '''
    function to tokenize the words and align the labels with the tokens
    '''

    tokenized_inputs = tokenizer(examples["words"], truncation=True, 
                                                    is_split_into_words=True, 
                                                    padding="max_length", 
                                                    max_length=args.max_len)
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

def do_collate(json_dicts):
        """
        collates example into a dict batch; produces ands pads tensors
        """

        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        batch["input_ids"] = pad_sequence(
            batch["input_ids"], padding_value=-100, batch_first=False
        )
        batch["attention_mask"] = pad_sequence(
            batch["attention_mask"], padding_value=-100, batch_first=False
        )
        batch["labels"] = pad_sequence(
            batch["labels"], padding_value=-100, batch_first=False
        )
        return dict(batch)

metric = load_metric("seqeval")
def compute_metrics(preds, labels):
    '''
    function to compute the metrics
    '''

    preds = preds.argmax(-1)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    return metric.compute(predictions=true_predictions, references=true_labels, zero_division = 1e-6)

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
    for split in ['train', 'test', 'valid']:
        tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        print(len(tokenized_datasets[split]))
    train_loader = DataLoader(tokenized_datasets['train'], collate_fn=do_collate, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(tokenized_datasets['valid'], collate_fn=do_collate, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_datasets['test'], collate_fn=do_collate, batch_size=args.batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


def train(args):
    '''
    function to train the model
    '''
    ## set tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ## load data
    dataset_dict = load_data(args)
    train_loader, valid_loader, test_loader = preprocess_data(args, dataset_dict, tokenizer)

    ## wandb
    if args.wandb:
        wandb.init(project='vanila_ner', name=args.model_name, config=args, save_code = True)
        wandb.watch(model)

    ## train and eval loop
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        torch.save(model, os.path.join(args.save_dir, f'epoch{epoch}.pt'))
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Epoch {:1d}, train".format(epoch), leave=False, disable=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits.view(-1, len(label_list)), labels.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": "{:.3f}".format(total_loss / (step + 1))})
            if step % 50 == 0:
                current_result = compute_metrics(outputs.logits, labels)
                print(current_result)
                if args.wandb: wandb.log({'train/loss': total_loss / (step + 1),'train/f1': current_result['overall_f1'],
                                        'train/precision': current_result['overall_precision'], 
                                        'train/recall': current_result['overall_recall'],
                                        'train/accuracy': current_result['overall_accuracy']})

        model.eval()
        total_loss = 0
        progress_bar = tqdm(valid_loader, desc="Epoch {:1d}, evaluation".format(epoch), leave=False, disable=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits.view(-1, len(label_list)), labels.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                current_result = compute_metrics(outputs.logits, labels)
                print(current_result)
                if args.wandb: wandb.log({'eval/loss': total_loss / (step + 1),'eval/f1': current_result['overall_f1'],
                                        'eval/precision': current_result['overall_precision'], 
                                        'eval/recall': current_result['overall_recall'],
                                        'eval/accuracy': current_result['overall_accuracy']})
        torch.save(model, os.path.join(args.save_dir, f'epoch{epoch}.pt'))

def main(args):
    train(args)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)