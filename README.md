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
| `convert` | Convert the data to spaCy's binary format |
| `train-base` | Train the NER model |
| `train-with-vec` | Train the NER model with vectors |
| `train-trf` | Train the NER model with a transformer |
| `train-trf-with-vec` | Train the NER model with a transformer with vectors |
| `evaluate-base` | Evaluate the base model and export metrics |
| `evaluate-with-vec` | Evaluate the model with vectors and export metrics |
| `evaluate-trf` | Evaluate the transformer model and export metrics |
| `evaluate-trf-with-vec` | Evaluate the transformer model with vec and export metrics |
| `visualize-model` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `base` | `convert` &rarr; `train-base` &rarr; `evaluate-base` |
| `vectors` | `convert` &rarr; `train-with-vec` &rarr; `evaluate-with-vec` |
| `transformers` | `convert` &rarr; `train-trf` &rarr; `evaluate-trf` |
| `transformers_vectors` | `convert` &rarr; `train-trf-with-vec` &rarr; `evaluate-trf-with-vec` |

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

## [Optional]
Create new python virtual environment:
```
$ python3 -m venv ~/envs/jutro-ner
```
And add the following line to your aliases (default `~/.bashrc`)
```
alias jutro-ner="source ~/envs/jutro-ner/bin/activate"
```
Now, to activate the virtual environment you just have to type jutro-ner.

## Install dependencies
```
$ cd jutro-ner
$ pip install -r requirements.txt
```

## Set up wandb
This project uses logging with [wandb.ai](https://wandb.ai/).
```
$ wandb login
```
And then paste the API key from wandb.ai

## Env
In order for `import-doccano` to work, you have to create [.env](./.env) file with your doccano credentials in the root directory.
Example [.env]() file:
```
DOCCANO_USERNAME=admin
DOCCANO_PASSWORD=password
```
