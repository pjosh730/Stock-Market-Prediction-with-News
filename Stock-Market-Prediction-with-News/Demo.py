#!/usr/bin/env python
# coding: utf-8
"""
Weikun Hu

#file_name = '../Data/dailynews.csv'
"""
import re
import pickle
import pandas as pd
#import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import model_from_json

def load_data(file_name):
    """
    # ## Load Data
    """
    # Download newsheadline from Reddit [https://www.reddit.com/r/worldnews/?hl=]
    #file_name = '../Data/dailynews.csv'
    data = pd.read_csv(file_name, header=None)
    data = data[:1]
    # ## Preprocessing
    # ### Raw text
    headlines = []
    for row in range(0, len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[row, 0:24]))
    string = ''.join(headlines)
    return string#print('raw data: ', string)


def clean_data(raw_text):
    """
    Data Cleaning
    """
    string = raw_text
    string = string.lower()
    string = re.sub(r'[^\w\s]', ' ', string) # remove punctuation
    string = ' '.join([w for w in string.split() if len(w) >= 3])
    return string
    # ## News Headline Visulizaiton

    # In[5]:
def visualization(clean_text):
    """
    visualization
    """
    stop1 = stopwords.words("english")
    stop = stop1
    stop_words = set(stop)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(clean_text)

    return wordcloud

#tfidf_model_name = "../Model/tfidf1.pkl"
def vectorization(tfidf_model_name, clean_text):
    """
    vectorization
    """
    tfidf = pickle.load(open(tfidf_model_name, 'rb'))
    vectorizer = TfidfVectorizer(min_df=0.04, max_df=0.3, max_features=200000,
                                 ngram_range=(2, 2), vocabulary=tfidf.vocabulary_)


    # ### Vectorize Newsheadlines
    x_tfidf = vectorizer.fit_transform([clean_text])
    x_tfidf = x_tfidf.toarray()
    return x_tfidf


    # ## Prediction

    # ### Load Deep learning model
# model_name1 = './Model/model.json'
# model_name2 = "./Model/model.h5"
def prediction_deep_learning(model_name1, model_name2, x_tfidf):
    """
    Preidict New Data
    """
    json_file = open(model_name1, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name2)


    # ### Predict DJIA Increase [1] or Decrease [0]

    return loaded_model.predict_classes(x_tfidf, verbose=0)
