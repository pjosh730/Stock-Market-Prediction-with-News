import re
import pandas as pd
import numpy as np

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

class RandomForestModel(object):
    """
    This file describes the random forest model built for classifying DJIA index changes
    """

    def __init__(self):
        """initializes the random forest model
        """
        df = pd.read_csv('../Data/Combined_News_DJIA.csv')
        self.data = df 

        self.train = self.data[self.data['Date'] < '2015-01-01']
        self.test = self.data[self.data['Date'] > '2014-12-31']    
        #data cleaning
        self.trainheadlines = []
        for row in range(0,len(self.train.index)):
            self.trainheadlines.append(' '.join(str(x) for x in self.train.iloc[row,2:27]))
        self.testheadlines = []
        for row in range(0,len(self.test.index)):
            self.testheadlines.append(' '.join(str(x) for x in self.test.iloc[row,2:27]))
        self.count = CountVectorizer()
        self.basictrain = self.count.fit_transform(self.trainheadlines)
        self.basictest = self.count.transform(self.testheadlines)
        self.tfidf1 = TfidfVectorizer( min_df=0.01, max_df=0.99, max_features = 200000, ngram_range = (1, 1))
        self.tfidf2 = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
        self.model = RandomForestClassifier()
        
    def count_vect_mod(self):
        count_fit = self.model.fit(self.basictrain,self.train['Label'])
        pred_countvect = count_fit.predict(self.basictest)
        acc_count = accuracy_score(self.test['Label'],pred_countvect)
        return self.basictrain.shape, self.basictest.shape, acc_count
    
    def tfidf_rf_mod1(self):
        tfidf_transform_train = self.tfidf1.fit_transform(self.trainheadlines)
        model_fit = self.model.fit(tfidf_transform_train, self.train['Label'])
        tfidf_transform_test = self.tfidf1.transform(self.testheadlines)
        pred = model_fit.predict(tfidf_transform_test)
        acc = accuracy_score(self.test['Label'],pred)
        return acc 
        
    def tfidf_rf_mod2(self):
        tfidf_transform_train2 = self.tfidf2.fit_transform(self.trainheadlines)
        model_fit2 = self.model.fit(tfidf_transform_train2, self.train["Label"])
        tfidf_transform_test2 = self.tfidf2.transform(self.testheadlines)
        pred2 = model_fit2.predict(tfidf_transform_test2)
        acc2 = accuracy_score(self.test['Label'],pred2)
        return acc2 
        
        
        
    
        
    
