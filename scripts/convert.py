"""Convert entity annotation from doccano-jsonl to spacy V3 .spacy format"""
import os
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import spacy
import typer
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from thinc.api import Config

from scripts.functions import make_customize_tokenizer

labels = set([
    "occurs on",
    "value",
])


def convert(df: pd.DataFrame, output_path: Path, ents_subset: List[str]):
    Doc.set_extension("rel", default={}, force=True)

    nlp = spacy.blank('pl')
    db = DocBin(store_user_data=True)

    make_customize_tokenizer()(nlp)

    for text, ents, relations, idx in zip(df['text'], df['entities'], df['relations'], df['id']):
        text = text.lower()
        doc = nlp.make_doc(text)

        ent_id_to_start_offset = {}
        rels = {}

        if ents_subset:
            ents = [ent for ent in ents if ent.get('label') in ents_subset]

        for ent in ents:
            start, end, label, ent_id = ent.get('start_offset'), ent.get(
                'end_offset'), ent.get('label'), ent.get('id')

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
                ent_id_to_start_offset[ent_id] = cs.start
                doc.set_ents([cs], default="unmodified")

        for rel in relations:
            from_id, to_id, rel_label = rel.get('from_id'), rel.get('to_id'), rel.get('type')

            if ent_id_to_start_offset.get(from_id) and ent_id_to_start_offset.get(to_id) and rel_label in labels:
                if not rels.get((from_id, to_id)):
                    rels[(from_id, to_id)] = {
                        rel_label: 1.,
                    }
                else:
                    rels[(from_id, to_id)][rel_label] = 1.

        for label in labels:
            for rel in rels.values():
                if label not in rel.keys():
                    rel[label] = 0

        doc._.rel = rels
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
