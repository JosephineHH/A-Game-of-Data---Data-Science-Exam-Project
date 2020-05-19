import pandas as pd

import numpy as np 
import re
import os 
import string
from string import punctuation
import _collections
from _collections import defaultdict
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
import collections

df = pd.read_csv('C:/Users/hille/Desktop/Data science/Project/A-Game-of-Data---Data-Science-Exam-Project/got_cleaned.csv', encoding = 'utf-8')


#Topic modelling
text = df['tokens']

dictionary = corpora.Dictionary(text.str.split())
corpus = [dictionary.doc2bow(txt) for txt in text.str.split()]


lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)

lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics
lsitopics = lsimodel.show_topics(formatted=False)

"""
#most used words
tfidf = models.TfidfModel(corpus, smartirs='ntc')
for doc in tfidf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

corpus_tfidf = tfidf[corpus]
print(corpus_tfidf)
# Human readable format of corpus (term-frequency)
[[(dictionary[id], freq) for id, freq in cp] for cp in corpus_tfidf[:1]]
#Prints score for the first sentence
"""
#-------------------------------------------------------------------
#term frequency
allWords = [sentence for sentence in df['lemma']]
allWords = [x for x in allWords if str(x) != 'nan']
allWords = ' '.join(allWords)

allWords = [word for word in allWords.split(' ')]


frequencyDist = nltk.FreqDist(allWords)
frequencyDist.most_common(50) #50 most common words

#Create df
pd.DataFrame(frequencyDist.items())

rslt = pd.DataFrame(frequencyDist.most_common(8000),
                    columns=['Word', 'Frequency']).set_index('Word')

#rslt.to_csv('C:/Users/hille/Desktop/Data science/Project/A-Game-of-Data---Data-Science-Exam-Project/50most_common.csv')

#-------------------------------------------------------------------
#Term frequency inverse document frequency 
#document = Episode
#Create documents for each season
documentDict = {}
bag_of_words = []

for i in range(1,len(set(df['N_serie']))+1):
    #Choose data from episode we are working with
    tempdf = df.loc[df['N_serie'] == i]

    tempallWords = [sentence for sentence in tempdf['lemma']]
    tempallWords = [x for x in tempallWords if str(x) != 'nan']
    tempallWords = ' '.join(tempallWords)
    tempallWords = [word for word in tempallWords.split(' ')]

    normalizedtf = {}

    tempDict = collections.Counter(tempallWords)
    for word, countw in tempDict.items():
        normalizedtf[word] = countw/len(tempallWords)
    
    for word in set(allWords):
        if word in set(tempallWords):
            pass
        else:
            normalizedtf[word] = 0


    documentDict[i] = normalizedtf
    print(i)

#Now that we have computed the term frequency we want to compute the inverse document frequency
import math
j = 1
idf = {}
for word in allWords:
    no_episodes_with_term = 0
    for i in range(1,len(set(df['N_serie']))+1):
        tempdf = df.loc[df['N_serie'] == i]

        tempallWords = [sentence for sentence in tempdf['lemma']]
        tempallWords = [x for x in tempallWords if str(x) != 'nan']
        tempallWords = ' '.join(tempallWords)
        tempallWords = [word for word in tempallWords.split(' ')]

        if word in tempallWords:
            no_episodes_with_term = no_episodes_with_term + 1
    
    idf_val = math.log(float(len(set(df['N_serie']))) / no_episodes_with_term) 
    
    idf[word] = idf_val
    j+=1
    print(j)




"""
#Find tfidf
id2word = corpora.Dictionary(document)

corpus = [id2word.doc2bow(txt) for txt in document]
print(corpus[:5])

tfidf = models.TfidfModel(corpus, smartirs='ntc')
for doc in tfidf[corpus]:
    print([[id2word[id], np.around(freq, decimals=2)] for id, freq in doc])

corpus_tfidf = tfidf[corpus]
print(corpus_tfidf)
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus_tfidf[:1]]
"""


#Try something else
#https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

bag_of_words = allWords
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, countw in wordDict.items():
        tfDict[word] = countw/float(bowCount)
        
    return tfDict

computeTF(documentDict,bag_of_words)