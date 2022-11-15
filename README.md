# cicero-nlp

## Training Model on Spacy

For our practice of trianing, there are four components needed for training a spacy model:For oThere are four components needed for training a spacy model

1. base_config.cfg
2. config.cfg
3. train.spacy
4. dev.spacy

### base_config.cfg

`base_config.cfg` is the blueprint of the config files that store all the settings and hyperparameters for training. It is used to generate the real config files, `config.cfg`, by using terminal commands.

We have two types of base_config.cfg — one is for light spacy model, the another one is for spacy transformer model.

The light cpu base_config.cfg can be found at:

The transformer gpu base_config.cfg can be found at: https://gist.github.com/ravenouse/bfea947bf69f59ec4de598de1879b678

You can create and specify your own base_config.cfg file at the [spaCy&#39;s offical training information page](https://spacy.io/usage/training).

#### Monitor Training with wandb

To have a better track of your training results and increase the reproducibility of your experiments, you may want to use [wandb](https://wandb.ai/site) to monitor, visulize, and record your training.

You can add the fellowing codes at the very end of the `base_config.cfg file` to achieve the functionality.

```
@loggers = "spacy.WandbLogger.v3"
project_name = "spacy_merge_model"
model_log_interval = 1000
remove_config_values = []
log_dataset_dir = null
entity = null
run_name = null
```

Our sample base_config file with those lines added can be found at:

You can refer to this wandb's [article](https://wandb.ai/wandb/wandb_spacy_integration/reports/Reproducible-spaCy-NLP-Experiments-with-Weights-Biases--Vmlldzo4NjM2MDk) and [colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/spacy/SpaCy_v3_and_W%26B.ipynb#scrollTo=QT5YtRqQN6aX) for futher detials.

### config.cfg

`config.cfg` is the real configuretion file used to specify training settings and hyperparameters. It is generated from `base_config.cfg` by using the below command.

```
!python -m spacy init fill-config ./base_config.cfg ./config.cfg
```

You can overwirte some settings or hyperparameters when you enter the run command to start the training, which will be brefiy introduced in the training section.

Our sample config file can be found at:

### train.spacy and dev.spacy

The `train.spacy` and `dev.spacy` are the data for training and evaluating. The files with `spacy` extension is compilable with training in spaCy and spaCy models. The detialed process of creating the data can be found in our notebooks. In short, the spacy file can be created and read as fellowing.

```

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
