# src/text_utils.py

import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK packages are downloaded on Streamlit Cloud
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def sentence_vector(sentence, w2v_model):
    words = word_tokenize(sentence.lower())
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)
