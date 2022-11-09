#!/usr/bin/env bash

for tag in spacy-cpu0 spacy-gpu0
do
    docker pull $1:$tag
    docker tag $1:$tag azavea/cicero:$tag 
done
