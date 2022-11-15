#!/usr/bin/env bash

echo "Input from: $1"
echo "Output to: $2"
aws s3 sync $1 /train-on-batch
python -m spacy train /train-on-batch/config.cfg --output /train-on-batch/output --paths.train /train-on-batch/train.spacy --paths.dev /train-on-batch/dev.spacy --gpu-id 0
aws s3 sync /train-on-batch/wandb $2/wandb
aws s3 sync /train-on-batch/output $2/$(date +"%Y%m%dT%H%M%S")/
