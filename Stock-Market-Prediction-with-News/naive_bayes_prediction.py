
"""
This file describes the naive bayes model used to predict DJIA price increase (1) and descrease (0)
"""
import os
import re
import nltk
import pandas as pd
import numpy as np
import pickle 

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

class NaiveBayesModel:
    """
    this is naive bayes model object using news data to predict stock trend
    """

    def __init__(self):
        """
        initialize the naive bayes model
        """
        # import the data; then split the data between testing and training sets based on date
        self.data = pd.read_csv('../Data/Combined_News_DJIA.csv')

        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']
            
        #data cleaning
        self.trainheadlines = []
        for row in range(0,len(self.train.index)):
            self.trainheadlines.append(' '.join(str(x) for x in self.train.iloc[row,2:27]))
        self.string_train_head  = ''.join(self.trainheadlines)

        self.testheadlines = []
        for row in range(0,len(self.test.index)):
            self.testheadlines.append(' '.join(str(x) for x in self.test.iloc[row,2:27]))
        self.string_test_head = ''.join(self.testheadlines)
        
        self.model=MultinomialNB(alpha=0.01)
        self.tfidf = TfidfVectorizer(min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))
        self.tfidf_train = self.tfidf.fit_transform(self.trainheadlines) 
        self.tfidf_test = self.tfidf.transform(self.testheadlines) 
        self.acc = []


    
    english_stemmer=nltk.stem.SnowballStemmer('english')


    
    def word_embedding(self):
        """
        Word embedding for training and testing dataset using the TD-IDF vectorizer
        input:nothing
        output: TF-IDF processed dataset  
        """
        print(self.tfidf_train.shape)
        print(self.tfidf_test.shape)
        return self.tfidf_train.shape,self.tfidf_test.shape
        return "embedding finished"

    def model_building(self):
        """
        Use the multnomial naive bayes model for classification
        """
        self.model.fit(self.tfidf_train, self.train["Label"])
        self.preds = self.model.predict(self.tfidf_test)
        self.acc=accuracy_score(self.test['Label'], self.preds)
        # Save the model to disk 
        file = 'naive_bayes_model.sav'
        pickle.dump(advancedmodel,open(file,'wb'))
        print("this should work now")

    