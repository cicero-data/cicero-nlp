#!/usr/bin/env bash

for tag in $(docker images | grep 'azavea/cicero \+nlp' | sed 's,[^ ]*[ ]*,,' | cut -f1 -d' ')
do
    docker tag azavea/cicero:$tag $1:$tag
    docker push $1:$tag
done
