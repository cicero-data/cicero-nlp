# To build the docker images

## Do this once

```bash
docker build -t azavea/spacy-gpu -f Dockerfile.spacy-gpu .
docker build -t azavea/spacy-cpu -f Dockerfile.spacy-cpu .
```

## Do this frequently

```bash
docker build -t azavea/cicero-nlp-gpu -f Dockerfile.cicero-nlp-gpu .
docker build -t azavea/cicero-nlp-cpu -f Dockerfile.cicero-nlp-cpu .
```
