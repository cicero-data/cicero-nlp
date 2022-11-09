# To build the docker images

This repo stores the code and data to train spacy models with the docker.

## Dependencies

The docker is needed for running the experiment.

For instruction of installing docker, please refer to this [page](https://docs.docker.com/get-docker/).

## Quickstart

### Pull or build the base images

You can either pull the base images from ECR (_recommended_) or you can build them again (_not recommended_).

### Choice 1: Pull the base images from ECR (recommended)

Before proceeding, make sure that you have logged into ECR from docker by typing `aws ecr get-login ...` (older versions of the AWS CLI) or `aws ecr get-login-password ...` (newer versions of the AWS CLI).
If you have done this in the past and have not logged out since you did it `docker logout ...`, then you do not need to do it again.

Once you have logged into ECR via docker, do the following.
Run the script `pull_and_tag.sh` and give the ECR repository name as its argument.
```bash
./pull_and_tag.sh xxxxxxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/cicero
```

### Choice 2: Build the base images (not recommended)

The below commands will build images that installs all the necessary
packages needed for spaCy model trianing. The images will be used as the base to build up for the following steps, training models. These images are needed to run once.

```bash
docker build -t azavea/cicero:spacy-gpu0 -f Dockerfile.spacy-gpu .
docker build -t azavea/cicero:spacy-cpu0 -f Dockerfile.spacy-cpu .
```

### Build the images for training models. Do this frequently

The below commands will build images for training models.
These commands should be run frequently once you change your training config or training data.

```bash
docker build -t azavea/cicero:nlp-gpu0 -f Dockerfile.cicero-nlp-gpu .
docker build -t azavea/cicero:nlp-cpu0 -f Dockerfile.cicero-nlp-cpu .
```

### Tag and push the images to ECR

This step is only necessary if you want to use the images on AWS Batch.

Before proceeding, make sure that you have logged into ECR from docker by typing `aws ecr get-login ...` (older versions of the AWS CLI) or `aws ecr get-login-password ...` (newer versions of the AWS CLI).
If you have done this in the past and have not logged out since you did it `docker logout ...`, then you do not need to do it again.

Once you have logged into ECR via docker, do the following.
Run the `tag_and_push.sh` script to tag and push your newly made training images to ECR.
The script takes a single argument which is the name of your ECR repository.
```bash
./tag_and_push.sh xxxxxxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/cicero
```

### Run the container locally or on an instance with using Nvidia GPUs

The below command will the image we just built in a container and lead you inside the container.

```bash
docker run -ti --runtime=nvidia -v <where you save your training files>:/train --entrypoint bash azavea/cicero:nlp-gpu0
```

In the container, run:

```bash
python -m spacy train <path to the config file> --output <path to save models> --paths.train <path to the train set> --paths.dev <path to the dev set> --gpu-id 0
```
