#!/usr/bin/env bash

echo "Starting spacy-wrapper-gpu.sh"
curl $5 > /train/file.zip
unzip /train/file.zip -d /train

python -m spacy train $1 --output /train/output --paths.train $2 --paths.dev $3 --gpu-id 0
aws s3 sync /train/output s3://raster-vision-mcclain/spacy_train/$4/$(date +"%Y%m%dT%H%M%S")/
