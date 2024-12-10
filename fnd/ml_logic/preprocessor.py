from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import time
import numpy as np
import pandas as pd

# Tokenise and remove stop words
def delete_stop_words(X):
    stop_words = set(stopwords.words('english'))
    return [word for word in X if word.lower() not in stop_words and word not in string.punctuation]

# apply Lemmatization-Funktion
def lemmatize_text(X):
    lemmatizer = WordNetLemmatizer()
    # Verben lemmatisieren
    verb_lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in X]
    # Nomen lemmatisieren
    return [lemmatizer.lemmatize(word, pos="n") for word in verb_lemmatized]

# NLP-Cleaning funktion
def nlp_cleaning(X):
    # Tokenisation
    X = X.apply(word_tokenize)
    # Remove stop words
    X = X.apply(delete_stop_words)
    # Lemmatisation
    X = X.apply(lemmatize_text)
    # Combine words into a string
    return X.apply(lambda x: ' '.join(x))

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    # start preprocessing data for nlp
    s = time.time()
    X = nlp_cleaning(X)
    print(f"Time to nlp clean: {time.time() - s:.2f} seconds")
    return X
