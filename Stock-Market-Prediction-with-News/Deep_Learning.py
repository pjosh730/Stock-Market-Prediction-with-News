#!/usr/bin/env python
# coding: utf-8
# Weikun Hu
# 2019/12/08
"""
Deep Learning Model
"""


import pickle
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
#import matplotlib.pyplot as plt


class DeepLearning:
    """
    This i Deep Leaning Model Class
    Train Model
    Make Prediction Using Model
    """
    def __init__(self):

        self.train_file_name = './Data/Combined_News_DJIA.csv'
        self.new_data_name = './Data/dailynews.csv'
        self.tfidf_model_name = "./Model/tfidf_DL.pkl"
        self.dl_model_name1 = './Model/dl_model.json'
        self.dl_model_name2 = './Model/dl_model.h5'

        #self.nb_model_name = './Model/naive_bayes_model.sav'
        self.tfidf = TfidfVectorizer(min_df=0.1, max_df=0.7,
                                     max_features=200000, ngram_range=(1, 1))
        #self.model = MultinomialNB(alpha=0.01)

    def train_dl_model(self):
        """
        Train Deep Learning Model
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

        advancedvectorizer = TfidfVectorizer(min_df=0.04, max_df=0.3,
                                             max_features=200000, ngram_range=(2, 2))
        advancedtrain = advancedvectorizer.fit(trainheadlines)
        pickle.dump(advancedtrain, open(self.tfidf_model_name, "wb"))
        advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
        y_train = np.array(train["Label"])
        y_test = np.array(test["Label"])


        ### LSTM
        max_features = 10000
        maxlen = 200
        batch_size = 32
        nb_classes = 2

        # vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(nb_words=max_features)
        tokenizer.fit_on_texts(trainheadlines)
        sequences_train = tokenizer.texts_to_sequences(trainheadlines)
        sequences_test = tokenizer.texts_to_sequences(testheadlines)

        X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)


        #print('Build model...')
        model = Sequential()
        model.add(Embedding(max_features, 128, dropout=0.2))
        model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=3,
                  validation_data=(X_test, Y_test))

        model_json = model.to_json()
        with open(self.dl_model_name1, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(self.dl_model_name2)

    def new_data_prediction(self):
        """
        Make prediction given new data
        """
        # Load Data
        df = pd.read_csv(self.new_data_name, header=None)
        df = df[:1]

        # Preprocessing Data

        headlines = []
        for row in range(0, len(df.index)):
            headlines.append(' '.join(str(x) for x in df.iloc[row, 0:24]))
        string = ''.join(headlines)

        string = string.lower()
        string = re.sub(r'[^\w\s]', ' ', string) # remove punctuation
        string = ' '.join([w for w in string.split() if len(w) >= 3])

        #Word Embedding
        #Load TFIDF Model
        tfidf = pickle.load(open(self.tfidf_model_name, 'rb'))
        vectorizer = TfidfVectorizer(min_df=0.04, max_df=0.3, max_features=200000,
                                     ngram_range=(2, 2), vocabulary=tfidf.vocabulary_)
        #Vectorize Newsheadlines

        x_tfidf = vectorizer.fit_transform([string])
        x_tfidf = x_tfidf.toarray()

        # Prediction
        # Load Deep learning model

        json_file = open(self.dl_model_name1, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.dl_model_name2)

        #Predict DJIA Increase [1] or Decrease [0]

        return loaded_model.predict_classes(x_tfidf, verbose=0)
