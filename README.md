# Cicero-NLP

![projectStructure](image/README/projectStructure.png)

## The Cicero Database

The [Cicero database](https://cicero.azavea.com/docs/) is a comprehensive, highly-accurate database of elected officials and legislative districts in 9 countries worldwide. It is made from public information that can be found on official government websites, and the information is organizedwell-separated into different categories. Those categories include names, phone numbers, fax numbers, and other contact information plus social media identifiers. The source URLs from official government websites are also incorporated into the dataset. Historically, this large dataset of over 57,000 historical and current officials was collected and managed via  extensive human annotation and data entry efforts. A sample of the Cicero Database is shown below.

![ciceroSample](image/README/ciceroSample.png)

## Data Processing

We have conduct three steps of data processing to make the Cicero Database ready to use:

1. Scrape origin webpages

   This step is to scrape politcans' webpages using the url address stored in the Cicero Dataset, and to save those webpages into HTML files.
2. Clean web pages and convert them into pure texts.

   This step is to clean the reduant HTML elements like HTML tags in the webpages, to transfer the content in the webpages into pure texts.
3. BIO tagging.

   This step is to BIO tag the interested information in the pure texts. This step will generate `train.spacy` and `dev.spacy`, which will be used as training set and developing set.

The codes for conducting these three steps can be found in the [developing_codes](/developing_codes).

## Training Models with spaCy

There are four components needed for training an Name Entity Recognition(NER) model using [spaCy](https://spacy.io/) library.

1. base_config.cfg
2. config.cfg
3. train.spacy
4. dev.spacy

### base_config.cfg

`base_config.cfg` is the blueprint of the config files that store all the settings and hyperparameters for training. It is used to generate the real config files, `config.cfg`, by using terminal commands.

You can create and specify your own base_config.cfg file at the [spaCys offical training information page](https://spacy.io/usage/training).

#### Monitor Training with wandb

To have a better track of your training results and increase the reproducibility of your experiments, you may want to use [wandb](https://wandb.ai/site) to monitor, visulize, and record your training.

You can add the fellowing codes at the very end of the `base_config.cfg file` to achieve the functionality.

```
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "<your project name>"
model_log_interval = 1000
remove_config_values = []
log_dataset_dir = null
entity = null
run_name = null
```

Our sample base_config file with those lines added can be found at [here](/sample/config/base_config.cfg).

You can refer to this wandb's [article](https://wandb.ai/wandb/wandb_spacy_integration/reports/Reproducible-spaCy-NLP-Experiments-with-Weights-Biases--Vmlldzo4NjM2MDk) and [colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/spacy/SpaCy_v3_and_W%26B.ipynb#scrollTo=QT5YtRqQN6aX) for futher detials.

### config.cfg

`config.cfg` is the real configuretion file used to specify training settings and hyperparameters. It is generated from `base_config.cfg` by using the below command.

```
python -m spacy init fill-config ./base_config.cfg ./config.cfg
```

Our sample config file can be found at [here](/sample/config/config.cfg).

### train.spacy and dev.spacy

The `train.spacy` and `dev.spacy` are the data for training and evaluating. The files with `spacy` extension is compilable with training in spaCy and spaCy models. The detialed process of creating these two files can be found in our codes of BIO tagging.

In short, the spacy file can be created and read as fellowing.

```python

# reference: https://spacy.io/api/docbin
# prepare
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
training_data = [
  ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
]

# write
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./train.spacy")

# read
db_read = DocBin().from_disk('./train.spacy')
db_read.strings


```

### Start training

Once you have `config.cfg`, `train.spacy`, and `dev.spacy`, you can start the training by using the below command.

```
python -m spacy train config.cfg --output ./ --paths.train /content/train.spacy --paths.dev /content/dev.spacy
```
