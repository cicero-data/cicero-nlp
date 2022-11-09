# To build the docker images

This repo stores the code and data to train spacy models with the docker.

## Dependencies

The docker is needed for running the experiment.

For instruction of installing docker, please refer to this [page](https://docs.docker.com/get-docker/).

## Quickstart

### Build the images for the environment. Do this once

The below commands will build images that installs all the necessary
packages needed for spaCy model trianing. The images will be used as the base to build up for the following steps, training models. These images are needed to run once.

```bash
docker build -t azavea/spacy-gpu -f Dockerfile.spacy-gpu .
docker build -t azavea/spacy-cpu -f Dockerfile.spacy-cpu .
```

### Build the images for training models. Do this frequently

The below commands will build images for training models. These commands should be run frequently once you change your training config or training data.

```bash
docker build -t azavea/cicero-nlp-gpu -f Dockerfile.cicero-nlp-gpu .
docker build -t azavea/cicero-nlp-cpu -f Dockerfile.cicero-nlp-cpu .
```

### Run the container with using Nvidia GPUs

The below command will the image we just built in a container and lead you inside the container.

```bash
docker run -ti --runtime=nvidia -v <where you save your training files>:/train --entrypoint bash azavea/cicero-nlp-gpu
```


In the container, run:

```bash
python -m spacy train <path to the config file> --output <path to save models> --paths.train <path to the train set> --paths.dev <path to the dev set> --gpu-id 0
```
