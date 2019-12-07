#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:

#file_name = '../Data/dailynews.csv'

import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import model_from_json

def load_data(file_name):

    # ## Load Data

    # Download newsheadline from Reddit [https://www.reddit.com/r/worldnews/?hl=]

    # In[2]:

    #file_name = '../Data/dailynews.csv'
    df = pd.read_csv(file_name, header=None)
    df = df[:1]
    #df


    # ## Preprocessing

    # ### Raw text

    # In[3]:


    headlines = []
    for row in range(0, len(df.index)):
        headlines.append(' '.join(str(x) for x in df.iloc[row, 0:24]))
    string = ''.join(headlines)
    return string
    
    #string


    # ### Clean Text 

    # In[4]:


    string = string.lower()
    string = re.sub(r'[^\w\s]', ' ', string) # remove punctuation
    string = ' '.join([w for w in string.split() if len(w) >= 3])
    #string


    # ## News Headline Visulizaiton

    # In[5]:


    stop1 = stopwords.words("english")
    stop = stop1
    stop_words = set(stop)
    wordcloud = WordCloud(width=800, height=800,
                    background_color='white',
                    stopwords=stop_words, 
                    min_font_size=10).generate(string)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()


    # ## Word Embedding

    # ### Load TFIDF Model

    # In[6]:


    tfidf = pickle.load(open("../Model/tfidf1.pkl", 'rb'))
    vectorizer = TfidfVectorizer(min_df=0.04, max_df=0.3, max_features=200000,
                                         ngram_range=(2, 2), vocabulary=tfidf.vocabulary_)


    # ### Vectorize Newsheadlines

    # In[7]:


    X_tfidf = vectorizer.fit_transform([string])
    X_tfidf = X_tfidf.toarray()
    X_tfidf


    # ## Prediction

    # ### Load Deep learning model

    # In[8]:


    json_file = open('../Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../Model/model.h5")


    # ### Predict DJIA Increase [1] or Decrease [0]

    # In[9]:


    print("Prediction Result for DJIA", loaded_model.predict_classes(X_tfidf, verbose=0))

