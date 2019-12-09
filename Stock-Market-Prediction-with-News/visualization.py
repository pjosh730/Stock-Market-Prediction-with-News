#!/usr/bin/env python
# coding: utf-8
# Weikun Hu
"""
Visulization
"""
# ### Import library


import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
def visulization():
    """
    visulization
    """
    data = pd.read_csv('../Data/Combined_News_DJIA.csv')
    df_inc = data[data.Label == 1]
    # separate all DJIA scores that increased in the dataset
    df_decr = data[data.Label == 0]
    # separate out all DJIA scores that descreased in the dataset
    print(df_inc.shape)
    print(df_decr.shape)

    headlines_inc = []
    for row in range(0, len(df_inc.index)):
        headlines_inc.append(' '.join(str(x) for x in df_inc.iloc[row, 2:27]))
    string_inc = ''.join(headlines_inc)

    string_inc = string_inc.lower()
    string_inc = re.sub(r'[^\w\s]', ' ', string_inc) # remove punctuation
    string_inc = ' '.join([w for w in string_inc.split() if len(w) >= 3])

    stop1 = stopwords.words("english")
    stop2 = ['say', 'says', 'new']
    stop = stop1 + stop2
    stop_words = set(stop)
    wordcloud_inc = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stop_words,
                              min_font_size=10).generate(string_inc)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_inc, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()

    headlines_decr = []
    for row in range(0, len(df_decr.index)):
        headlines_decr.append(' '.join(str(x) for x in df_decr.iloc[row, 2:27]))
    string_decr = ''.join(headlines_decr)

    string_decr = string_decr.lower()
    string_decr = re.sub(r'[^\w\s]', ' ', string_decr) # remove punctuation
    string_decr = ' '.join([w for w in string_decr.split() if len(w) >= 3])

    stop1 = stopwords.words("english")
    stop2 = ['say', 'says', 'new']
    stop = stop1 + stop2
    stop_words = set(stop)
    wordcloud_decr = WordCloud(width=800, height=800,
                               background_color='black',
                               stopwords=stop_words,
                               min_font_size=10).generate(string_decr)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_decr, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
