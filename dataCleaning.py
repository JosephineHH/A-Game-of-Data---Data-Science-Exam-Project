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


df = pd.read_csv('C:/Users/hille/Desktop/Data science/Project/A-Game-of-Data---Data-Science-Exam-Project/got_half_cleaned.csv', encoding = 'unicode_escape', sep = ";")



#We start by defining lists of words to remove
my_stopwords = nltk.corpus.stopwords.words('english') #uninformative common words
my_punctuation = r'!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•…' #punctuation
#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
wordnet_lemmatizer = WordNetLemmatizer()



def clean_sentence(sentence, lemma=False):
    sentence = re.sub(r'[^\w\s]', ' ', sentence) # strip punctuation
    sentence = re.sub(r'\s+', ' ', sentence) #remove double spacing
    sentence_token_list = [word for word in sentence.split(' ')
                            if word not in my_stopwords] # remove stopwords

    if lemma == True:
      sentence_lemma_list = [wordnet_lemmatizer.lemmatize(word) if '#' not in word else word
                        for word in sentence_token_list] # apply lemmatizer
    else:   # or                 
      sentence_token_list = [word_rooter(word) if '#' not in word else word
                        for word in sentence_token_list] # apply word rooter
    
    #sentence = ' '.join(sentence_token_list)
    sentence_lemma = ' '.join(sentence_lemma_list)
    return sentence_lemma

#Finally we apply the function to clean tweets (here we use lemmas)
df['tokens'] = df.Sentence.apply(clean_sentence, lemma=False)


df['lemma'] = df.Sentence.apply(clean_sentence, lemma=True)



df['token_text'] = [
    [word for word in sentence.split()]
    for sentence in df['lemma']]
print(df['token_text'])

df.to_csv('C:/Users/hille/Desktop/Data science/Project/A-Game-of-Data---Data-Science-Exam-Project/got_cleaned.csv', index=False)