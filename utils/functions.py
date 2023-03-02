from spacy.util import registry, compile_suffix_regex, compile_infix_regex, compile_prefix_regex

@registry.callbacks("customize_tokenizer")
def make_customize_tokenizer():
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

    return customize_tokenizer