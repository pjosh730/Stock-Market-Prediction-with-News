
"""
This file describes the naive bayes model used to predict DJIA price increase (1) and descrease (0)
"""
import os
import re
import nltk
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # this is the library that assumes the likelihood function to have a multinomial distribution
from sklearn.metrics import accuracy_score

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer

class NaiveBayesModel(object):
    """
    this is naive bayes model object using news data to predict stock trend
    """

    def __init__(self):
        """
        initialize the naive bayes model
        """
        # import the data; then split the data between testing and training sets based on date
        df = pd.read_csv('../Data/Combined_News_DJIA.csv')
        self.data = df 

        self.train = df[df['Date'] < '2015-01-01']
        self.test = df[df['Date'] > '2014-12-31']
            
        #data cleaning
        self.trainheadlines = []
        for row in range(0,len(self.train.index)):
            self.trainheadlines.append(' '.join(str(x) for x in self.train.iloc[row,2:27]))
        self.string_train_head  = ''.join(self.trainheadlines)

        self.testheadlines = []
        for row in range(0,len(self.test.index)):
            self.testheadlines.append(' '.join(str(x) for x in self.test.iloc[row,2:27]))
        self.string_test_head = ''.join(self.testheadlines)

    
    english_stemmer=nltk.stem.SnowballStemmer('english')


    
    def word_embedding(self):
        """
        Word embedding for training and testing dataset using the TD-IDF vectorizer
        input:self
        output: TF-IDF processed dataset  
        """
        tfidf = TfidfVectorizer(min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))
        self. tfidf_train = tfidf.fit_transform(self.trainheadlines) 
        self.tfidf_test = tfidf.transform(self.testheadlines) 
        print(tfidf_train.shape)
        print(tfidf_test.shape)
        return self.tfidf_train,self.tfidf_test

    def model_building(self):
        """
        Use the multnomial naive bayes model for classification
        """
        advancedmodel = MultinomialNB(alpha=0.01)
        advancedmodel = advancedmodel.fit(self.tfidf_train, train["Label"])
        self.preds = advancedmodel.predict(self.tfidf_test)
        self.acc=accuracy_score(self.test['Label'], self.preds)

    