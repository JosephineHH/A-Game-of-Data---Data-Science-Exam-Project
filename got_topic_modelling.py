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
frequencyDist.most_common(21) #50 most common words

#Create df
pd.DataFrame(frequencyDist.items())

#-------------------------------------------------------------------
#Term frequency inverse document frequency 
#document = Episode
#Create documents for each season
document = []

for i in range(0,len(set(df['N_serie']))):
    #Choose data from episode we are working with
    tempdf = df.loc[df['N_serie'] == i]

    allWords = [sentence for sentence in df['Sentence']]
    allWords = [x for x in allWords if str(x) != 'nan']
    allWords = ' '.join(allWords)
    allWords = [word for word in allWords.split(' ')]


    document.append(allWords)
    print(i)

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



#Try something else
#https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count is wordDict.items():
        tfDict(word) = count/float(bowCount)
        
    return tfDict