# src/prediction.py

import joblib
from gensim.models import Word2Vec
import numpy as np
from src.text_utils import sentence_vector

# Load models once
svm_model = joblib.load("models/svm_model.pkl")
w2v_model = Word2Vec.load("models/word2vec.model")

def predict_text(text):
    vector = sentence_vector(text, w2v_model).reshape(1, -1)
    prediction = svm_model.predict(vector)[0]
    # Get probability of class 1 (Hate Speech)
    prob_hate = svm_model.predict_proba(vector)[0][1]
    return prediction, prob_hate
