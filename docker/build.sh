#!/usr/bin/env bash

docker build -t azavea/cicero:nlp-gpu0 -f Dockerfile.cicero-nlp-gpu .
docker build -t azavea/cicero:nlp-cpu0 -f Dockerfile.cicero-nlp-cpu .
