from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import time
import numpy as np
import pandas as pd
import re
import string
from params import *

def prepare_basic_clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    ##########################################################
    #                 Delete redundant column                #
    ##########################################################
    # delete redundant column 'Unnamed: 0'
    df.drop('Unnamed: 0', axis=1, inplace=True)

    ##########################################################
    #               Compressing Data                         #
    ##########################################################
    # Compressing datatypes: raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)
    ##########################################################
    #        Handling missing and redundant Data             #
    ##########################################################
    # reduce column title after combining
    # filling the missing data with spaces
    df = df.fillna(' ') ## aplying to na via fillna
    # combine the text and the title
    df['text'] = df['title'] + df['text']
    # delete title column, beacuse it's included in text column
    df.drop('title',axis=1, inplace=True)

    # Remove buggy transactions
    df.drop_duplicates(inplace=True)
    ##########################################################
    #                     check Balancing                    #
    ##########################################################
    # Calculate percentage of NaN values in each column
    # Balancing: Calculation of the distribution in per cent
    label_counts = df['label'].value_counts(normalize=True) * 100
    # Formatted output
    print(f"Balancing result is label 0: {label_counts.get(0, 0):.2f}%, label 1: {label_counts.get(1, 0):.2f}%")

    ##########################################################
    #                     NaN Handling                       #
    ##########################################################
    # Calculate percentage of NaN values in each column
    nan_percentage = (df.isna().sum() / len(df)) * 100
    # Format output with percentage symbol
    formatted_nan_percentage = nan_percentage.apply(lambda x: f"{x:.2f}%")
    # Print each column's NaN percentage
    for col, percentage in formatted_nan_percentage.items():
        print(f"{col}: {percentage}")
    # Remove buggy transactions
    df.dropna(how='any', axis=0, inplace=True)
    ##########################################################
    #                    basic cleaning                      #
    ##########################################################
    s = time.time()
    df = _basic_cleaning(df)
    print(df.info())
    time_to_clean = time.time() - s
    print('Time for basic cleaning {:.2f} s'.format(time_to_clean))

    print("✅ data cleaned")

    return df

def preprocess_features(X: pd.Series) -> pd.DataFrame:
    # start preprocessing data for nlp
    s = time.time()
    X = _nlp_cleaning(X)
    print(f"Time to nlp clean: {time.time() - s:.2f} seconds")
    return X

def _basic_cleaning(df):
    # stripping:
    df['text'] = df['text'].str.strip()

    # tolower:
    df['text'] = df['text'].str.lower()

    # digit: Remove digits from each row of the ‘text’ column
    df['text'] = df['text'].apply(lambda x: ''.join(char for char in x if not char.isdigit())).astype('string')

    # punctuation: Remove all punctuation marks from the ‘text’ column
    df['text'] = df['text'].str.replace(r'[{}]'.format(re.escape(string.punctuation)), '', regex=True)

    # delete html-tags
    df['text'] = df['text'].apply(lambda x: re.sub('<[^<]+?>', '', x)).astype('string')

    return df

# NLP-Cleaning funktion
def _nlp_cleaning(X: pd.Series) -> pd.DataFrame:
    # Tokenisation
    X = X.apply(word_tokenize)
    # Remove stop words
    X = X.apply(_delete_stop_words)
    # Lemmatisation
    X = X.apply(_lemmatize_text)
    # Combine words into a string
    return X.apply(lambda x: ' '.join(x))

# Tokenise and remove stop words
def _delete_stop_words(X):
    stop_words = set(stopwords.words('english'))
    return [word for word in X if word.lower() not in stop_words and word not in string.punctuation]

# apply Lemmatization-Funktion
def _lemmatize_text(X):
    lemmatizer = WordNetLemmatizer()
    # Verben lemmatisieren
    verb_lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in X]
    # Nomen lemmatisieren
    return [lemmatizer.lemmatize(word, pos="n") for word in verb_lemmatized]
