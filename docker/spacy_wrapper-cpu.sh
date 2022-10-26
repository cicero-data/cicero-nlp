#!/usr/bin/env bash

python -m spacy train $1 --output ${MODEL_DIR} --paths.train $2 --paths.dev $3
# aws s3 sync ${MODEL_DIR} s3://raster-vision-mcclain/spacy_test/$(date +"%Y%m%dT%H%M%S")/
