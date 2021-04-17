#437 Final Project - Twitter User Gender Classification by Tweet - Logan Clark, Puthypor Sengkeo, and Nick Lamos
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words

#Import data
#Data split is 6194 male - 6700 female

#Current Data Format:
#_unit_id | gender | text | tweet_count | description | name 
#description, name, and tweet_count are not used as of now, if we don't ever use them we can removed them from the data set
df = pd.read_csv('cleanedGenderClassifierData.csv', engine='python')

#Convert dataframe to numpy array
npArray = df.to_numpy()

xList = list()
yList = list()
#Split into X and y
for entry in npArray:
    yList.append(entry[1])
    xList.append(entry[2])
npY = np.array(yList)


#vectorizer for converting to lower case and removing punct
vectorizer = CountVectorizer()

#remove english stop words
vectorizer.set_params(stop_words='english')

#include 1-grams (1 grams performed best)
vectorizer.set_params(ngram_range=(1,1))

#fit vectorizer
vectorizedX = vectorizer.fit_transform(xList)

#normalize freq based on tweet length (tfidf)
tfidf_transformer = TfidfTransformer()
tfidfX = tfidf_transformer.fit_transform(vectorizedX)

#classify using naive Bayes
clf = MultinomialNB().fit(tfidfX, npY)

#cross validation
scores = cross_val_score(clf, tfidfX, yList, cv=3, scoring='accuracy')
f1Macro = cross_val_score(clf, tfidfX, yList, cv=3, scoring='f1_macro')

print('Number of tweets:', len(yList), ', 3-fold accuracy:', np.mean(scores), ', F1-Macro:', np.mean(f1Macro))
