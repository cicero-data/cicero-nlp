#!/usr/bin/env bash

python -m spacy train ./config.cfg --output ${MODEL_DIR} --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 0
aws s3 sync ${MODEL_DIR} s3://raster-vision-mcclain/spacy_test/$(date +"%Y%m%dT%H%M%S")/
