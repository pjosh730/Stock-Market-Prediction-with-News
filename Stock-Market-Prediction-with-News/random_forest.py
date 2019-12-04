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

		self.train = df[df['Date'] < '2015-01-01']
		self.test = df[df['Date'] > '2014-12-31']
			
		#data cleaning
		self.trainheadlines = []
		for row in range(0,len(self.train.index)):
			self.trainheadlines.append(' '.join(str(x) for x in self.train.iloc[row,2:27]))
		

		self.testheadlines = []
		for row in range(0,len(self.test.index)):
			self.testheadlines.append(' '.join(str(x) for x in self.test.iloc[row,2:27]))
		
	def count_vect(self):
		basicvectorizer = CountVectorizer()
		self.basictrain = basicvectorizer.fit_transform(self.trainheadlines)
		print(self.basictrain.shape)
		self.basictest = basicvectorizer.transform(self.testheadlines)
		print(self.basictest.shape)
		
