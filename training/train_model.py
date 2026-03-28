# training/train_model.py

import os
import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset
import nltk

# ----------------------------
# Setup
# ----------------------------

nltk.download("punkt")

print("Loading dataset...")

# Load dataset
ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
train_df = ds["train"].to_pandas()

# 🔥 Reduce dataset size for faster training
train_df = train_df.sample(n=20000, random_state=42)

# Convert to binary label
train_df["label"] = train_df["toxic"]

texts = train_df["comment_text"].astype(str)
labels = train_df["label"]

# ----------------------------
# Tokenization
# ----------------------------

print("Tokenizing text...")
sentences = [word_tokenize(text.lower()) for text in texts]

# ----------------------------
# Train Word2Vec
# ----------------------------

print("Training Word2Vec...")

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# ----------------------------
# Sentence to Vector
# ----------------------------

def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)

print("Creating feature matrix...")

X = np.array([sentence_vector(text) for text in texts])
y = labels.values

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Linear SVM (FAST)
# ----------------------------

print("Training Linear SVM...")

from sklearn.calibration import CalibratedClassifierCV

base_svm = LinearSVC()
svm_model = CalibratedClassifierCV(base_svm)

svm_model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------

y_pred = svm_model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------
# Save Models
# ----------------------------

os.makedirs("models", exist_ok=True)

w2v_model.save("models/word2vec.model")
joblib.dump(svm_model, "models/svm_model.pkl")

print("\nModels saved successfully.")