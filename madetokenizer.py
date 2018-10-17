import nltk

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktLanguageVars
from nltk.tokenize.treebank import TreebankWordTokenizer

import basic

class CustomSentenceBreakingLangVars(PunktLanguageVars):
    mything = 'something'
    # this does nothing -- these must be changed after construction
    #send_end_chars = ('.', '!')

def build_made_tokenizer(keep_token_strings = False):
    print('Building MADE tokenizer...')
    cs_preprocess_split_re_strings = []
    # double newlines
    cs_preprocess_split_re_strings.append(r'[\r\n]{2,}')
    # newlines with only spaces
    cs_preprocess_split_re_strings.append(r'[\r\n]+\s+[\r\n]+')
    # numbered lists (e.g. "1.", "2)")
    cs_preprocess_split_re_strings.append(r'(^|\r|\n)+\s*\d+[.)-]')
    # bulleted lists (e.g."*", "-")
    cs_preprocess_split_re_strings.append(r'(^|\r|\n)+\s*[*-]')
    # starting labels (e.g. "WEIGHT:")
    cs_preprocess_split_re_strings.append(r'(^|\r|\n)+\s*\w+[:]')
    # break up other lines separated by dates
    cs_preprocess_split_re_strings.append(r'(^|\r|\n)+\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
    # MIMIC has many lines that start with this [**YYYY-M-DD**]
    cs_preprocess_split_re_strings.append(r'^\[\*+\d{4}-\d{1,2}-\d{1,2}\*+\]')
    # TIU notes have long bars like this : '***********' or '===========' or '------'
    cs_preprocess_split_re_strings.append(r'[*=-]{3,}')
    
    # NOTE : This breaking rule was disabled 2-13-18 since the UMass MADE challenge data often ended each line with 2 spaces and a 
    # newline which caused this aggressive rule to fire over and over again.
    # aggressively break anything with lots of spaces (tabular data)
    #cs_preprocess_split_re_strings.append(r'\s{3,}')
    
        
    custom_lang_vars = CustomSentenceBreakingLangVars()
    custom_lang_vars.sent_end_chars = ('.', '!')
    print(custom_lang_vars.sent_end_chars)

    punkt_tokenizer2 =  PunktSentenceTokenizer(lang_vars = custom_lang_vars)
    treebank_tokenizer = TreebankWordTokenizer()

    # looks like "pt." and "D.R." and "P.R." are already being handled
    #punkt_tokenizer2._params.abbrev_types.update(extra_abbrev)    
        
    cs_tokenizer = basic.nlp.tokenizers.clinical_tokenizers.ClinicalSentenceTokenizer(default_sentence_tokenizer = punkt_tokenizer2, preprocess_split_re_strs = cs_preprocess_split_re_strings)

    made_index_tokenizer = basic.nlp.tokenizers.clinical_tokenizers.IndexTokenizer(cs_tokenizer, treebank_tokenizer, keep_token_strings = keep_token_strings)
    
    return made_index_tokenizer