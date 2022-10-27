#!/usr/bin/env bash

python -m spacy train $1 --output /train/output --paths.train $2 --paths.dev $3
aws s3 sync /train/output s3://raster-vision-mcclain/spacy_test/address/$(date +"%Y%m%dT%H%M%S")/