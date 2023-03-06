<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: ZPP Bachelor's project.

NER in a new pipeline (Named Entity Recognition) to work with medical documentation.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `import-doccano` | Export labeled data from doccano and unpack it in /assets |
| `download-lg` | Download a spaCy model with pretrained vectors |
| `pretrain` | Pretrain the vectors. |
| `convert` | Convert the data to spaCy's binary format |
| `debug-data` | Analyze and validate training and development data. |
| `debug-data-trf` | Analyze and validate training and development data for transformer config. |
| `train-base` | Train the NER model |
| `train-with-vec` | Train the NER model with vectors (bad, don't use) |
| `train-trf` | Train the NER model with a transformer |
| `evaluate-base` | Evaluate the base model and export metrics |
| `evaluate-with-vec` | Evaluate the model with vectors and export metrics |
| `evaluate-trf` | Evaluate the transformer model and export metrics |
| `visualize` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `base` | `debug-data` &rarr; `convert` &rarr; `train-base` &rarr; `evaluate-base` |
| `vectors` | `debug-data` &rarr; `convert` &rarr; `train-with-vec` &rarr; `evaluate-with-vec` |
| `transformers` | `debug-data-trf` &rarr; `convert` &rarr; `train-trf` &rarr; `evaluate-trf` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/all.jsonl`](assets/all.jsonl) | Local | All annotated data |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

# Local setup
## Clone
To start development, first clone the repository with
```
$ git clone git@github.com:ZPP-Jutro-2023/jutro-ner.git
```
Then you have to install all the packages.

## Install poetry
Poetry is a really clean dependency manager and is used in this project.

On how to install, refer to [Poetry](https://python-poetry.org/docs/).

## Install dependencies and hooks
```
$ poetry install [--with cuda | m1]
$ poetry run pre-commit install
```

## Set up wandb
This project uses logging with [wandb.ai](https://wandb.ai/).
```
$ poetry run wandb login
```
And then paste the API key from wandb.ai

## Env
In order for `import-doccano` to work, you have to create [.env](./.env) file with your doccano credentials in the root directory.
Example [.env]() file:
```
DOCCANO_USERNAME=admin
DOCCANO_PASSWORD=password
```
