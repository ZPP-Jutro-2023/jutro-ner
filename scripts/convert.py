"""Convert entity annotation from doccano-jsonl to spacy V3 .spacy format"""
import os
import warnings
from pathlib import Path

import pandas as pd
import spacy
import typer
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from spacy.util import (compile_infix_regex, compile_prefix_regex,
                        compile_suffix_regex)


def customize_tokenizer(nlp):
        infixes = nlp.Defaults.infixes + [r'''=''',r'''\+''', r'''\/''', r'''\)''', r'''\*''', r'''\.''']
        infix_regex = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        suffixes = nlp.Defaults.suffixes + [r'''%$''']
        suffix_regex = compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffix_regex.search

        prefixes = nlp.Defaults.prefixes + [r'''^-''']
        prefix_regex = compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

def convert(df: pd.DataFrame, output_path: Path):
    nlp = spacy.blank('pl')
    db = DocBin()

    customize_tokenizer(nlp)

    for text, ents, id in zip(df['text'], df['entities'], df['id']):
        text = text.lower()
        doc = nlp.make_doc(text)
        char_spans = []
        for ent in ents:
            start, end, label = ent.get('start_offset'), ent.get('end_offset'), ent.get('label')

            # getting rid of accidental spaces. I wanted to avoid using alignment mode, because it may cut more than we expect, this felt safer.
            if text[start] == ' ':
                start += 1
            if text[end - 1] == ' ':
                end -= 1

            cs = doc.char_span(start, end, label)

            if cs is None:
                msg = f"Skipping entity id: {id} [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                char_spans.append(cs)
            
            doc.set_ents(char_spans)
            db.add(doc)
        
        db.to_disk(output_path)

def create_train_dev(input_path: Path, output_path: Path, train_size: float):
    train_path = os.path.join(output_path, 'train.spacy')
    dev_path = os.path.join(output_path, 'dev.spacy')

    print("Reading jsonl file...")
    df = pd.read_json(input_path, lines=True)
    # split to train and dev
    train, dev = train_test_split(df, train_size=train_size, random_state=42)

    print("Creating train dataset...")
    convert(train, train_path)

    print("Creating validation dataset...")
    convert(dev, dev_path)


if __name__ == "__main__":
    typer.run(create_train_dev)
