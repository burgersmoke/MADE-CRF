import sys
import os

import numpy as np

# load GloVe embeddings which are a simple space-delimited file where each line is a word and then floating point values
def load_glove_embeddings(file_path):
    word2vector_map = {}
    file = open(file_path, 'r', encoding = 'utf8')
    for line in file:
        values = line.split()
        word = values[0]
        # grab the rest of the tokens as a floating point vector
        vector = np.asarray(values[1:], dtype = 'float32')
        word2vector_map[word] = vector
    file.close()
    return word2vector_map