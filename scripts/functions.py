from spacy.util import (compile_infix_regex, compile_prefix_regex,
                        compile_suffix_regex, registry)

from scripts.rel_model import (create_classification_layer,
                               create_instance_processor, create_instances,
                               create_relation_model, create_tensors)
from scripts.rel_pipe import make_relation_extractor


@registry.callbacks("customize_tokenizer")
def make_customize_tokenizer():
    def customize_tokenizer(nlp):
        infixes = nlp.Defaults.infixes + [r'''=''', r'''\+''', r'''\/''', r'''\)''', r'''\*''', r'''\.''']
        infix_regex = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        suffixes = nlp.Defaults.suffixes + [r'''%$''']
        suffix_regex = compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffix_regex.search

        prefixes = nlp.Defaults.prefixes + [r'''^-''']
        prefix_regex = compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

    return customize_tokenizer
