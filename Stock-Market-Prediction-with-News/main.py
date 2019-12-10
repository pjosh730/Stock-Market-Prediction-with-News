#!/usr/bin/env python
# coding: utf-8
# Weikun Hu


from deep_learning import DeepLearning

DL = DeepLearning()
# Train Deep Learning Model
DL.train_dl_model()
# Make new prediction using Deep Learning Model
DL.new_data_prediction()


from naive_bayes_model import NaiveBayesModel
NB = NaiveBayesModel()
# Train naive bayes model
NB.train_nb_model()
# Make prediction for new data
NB.new_data_prediction()



