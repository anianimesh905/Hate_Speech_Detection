# src/text_utils.py

import numpy as np
from nltk.tokenize import word_tokenize

def sentence_vector(sentence, w2v_model):
    words = word_tokenize(sentence.lower())
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)