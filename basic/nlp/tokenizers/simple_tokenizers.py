import re

# split on anything non-whitespace
WHITESPACE_EXPRESSION = r'[^\s]+'
WHITESPACE_TOKENIZER_RE = re.compile(WHITESPACE_EXPRESSION, re.UNICODE)

# this matches the definition in TensorFlow's text preprocessing (preprocessing/text.py)
def whitespace_tokenizer_fn(iterator):
    for value in iterator:
        yield WHITESPACE_TOKENIZER_RE.findall(value)