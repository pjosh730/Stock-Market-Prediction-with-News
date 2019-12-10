#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:50:51 2019

@author: ruiyuzeng

Update Dec 7
By Weikun Hu
This file describes the naive bayes model used to predict DJIA price increase (1) and descrease (0)
"""

import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class NaiveBayesModel:
    """
    this is naive bayes model object using news data to predict stock trend
    the class contains two funtions
    1. train deep learning model
    2. make prediction using pre-trained model
    """

    def __init__(self):
        """
        initialize the naive bayes model
        """
        self.train_file_name = '../Data/Combined_News_DJIA.csv'
        self.new_data_name = '../Data/dailynews.csv'
        self.tfidf_model_name = "../Model/tfidf_NB.pkl"
        self.nb_model_name = '../Model/naive_bayes_model.sav'
        self.tfidf = TfidfVectorizer(min_df=0.1, max_df=0.7,
                                     max_features=200000, ngram_range=(1, 1))
        self.model = MultinomialNB(alpha=0.01)

    def train_nb_model(self):
        """
        Word embedding for training and testing dataset using the TD-IDF vectorizer
        input:nothing
        output: TF-IDF processed dataset
        """

        data = pd.read_csv(self.train_file_name)
        train = data[data['Date'] < '2015-01-01']
        test = data[data['Date'] > '2014-12-31']

        trainheadlines = []
        for row in range(0, len(train.index)):
            trainheadlines.append(' '.join(str(x) for x in train.iloc[row, 2:27]))

        testheadlines = []
        for row in range(0, len(test.index)):
            testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

        tfidf = self.tfidf
        tfidf_save = tfidf.fit(trainheadlines)
        pickle.dump(tfidf_save, open(self.tfidf_model_name, "wb"))

        tfidf_train = tfidf.fit_transform(trainheadlines)
        tfidf_test = tfidf.transform(testheadlines)

        model = self.model.fit(tfidf_train, train["Label"])
        preds = model.predict(tfidf_test)
        acc = accuracy_score(test['Label'], preds)
        pickle.dump(model, open(self.nb_model_name, 'wb'))


    def new_data_prediction(self):
        """
        Make Prediction for New Data
        The result is either 1 (DJIA increases)
        Or 0 (DJIA decreses)
        """
        data = pd.read_csv(self.new_data_name, header=None)
        data = data[:1]

        headlines = []
        for row in range(0, len(data.index)):
            headlines.append(' '.join(str(x) for x in data.iloc[row, 0:24]))
        string = ''.join(headlines)
        # Clean Data

        string = string.lower()
        string = re.sub(r'[^\w\s]', ' ', string) # remove punctuation
        string = ' '.join([w for w in string.split() if len(w) >= 3])

        # Vectorization
        tfidf = pickle.load(open(self.tfidf_model_name, 'rb'))
        vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.7, max_features=200000,
                                     ngram_range=(1, 1), vocabulary=tfidf.vocabulary_)

        # self.tfidf = TfidfVectorizer(min_df=0.1, max_df=0.7,
        #                              max_features=200000, ngram_range=(1, 1))
        x_tfidf = vectorizer.fit_transform([string])
        x_tfidf = x_tfidf.toarray()

        # Prediction
        model = pickle.load(open(self.nb_model_name, 'rb'))

        return model.predict(x_tfidf)
