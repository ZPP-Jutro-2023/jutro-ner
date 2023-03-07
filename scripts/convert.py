"""Convert entity annotation from doccano-jsonl to spacy V3 .spacy format"""
import os
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import spacy
import typer
from functions import make_customize_tokenizer
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from thinc.api import Config


def convert(df: pd.DataFrame, output_path: Path, ents_subset: List[str]):
    nlp = spacy.blank('pl')
    db = DocBin()

    make_customize_tokenizer()(nlp)

    for text, ents, idx in zip(df['text'], df['entities'], df['id']):
        text = text.lower()
        doc = nlp.make_doc(text)

        if ents_subset:
            ents = [ent for ent in ents if ent.get('label') in ents_subset]

        for ent in ents:
            start, end, label = ent.get('start_offset'), ent.get(
                'end_offset'), ent.get('label')

            # getting rid of accidental spaces. I wanted to avoid using alignment mode,
            # because it may cut more than we expect, this felt safer.
            if text[start] == ' ':
                start += 1
            if text[end - 1] == ' ':
                end -= 1

            cs = doc.char_span(start, end, label)

            if cs is None:
                msg = f"Skipping entity id: {idx} [{start}, {end}, {label}] in the following text because the"\
                    " character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                doc.set_ents([cs], default="unmodified")

            db.add(doc)

        db.to_disk(output_path)


def create_train_dev(input_path: Path, output_path: Path, train_size: float, ents_cfg_path: Path):
    train_path = os.path.join(output_path, 'train.spacy')
    dev_path = os.path.join(output_path, 'dev.spacy')

    ents_config = Config().from_disk(ents_cfg_path)
    ents_subset = ents_config['ents_settings'].get('subset')

    print("Reading jsonl file...")
    df = pd.read_json(input_path, lines=True)
    # split to train and dev
    train, dev = train_test_split(df, train_size=train_size, random_state=42)

    print("Creating train dataset...")
    convert(train, train_path, ents_subset)

    print("Creating validation dataset...")
    convert(dev, dev_path, ents_subset)


if __name__ == "__main__":
    typer.run(create_train_dev)
