from gensim.corpora import Dictionary
from gensim import corpora, models
# import libraries
import pandas as pd
import numpy as np
import pickle
import os
import re
from konlpy.tag import Okt
okt = Okt()
from konlpy.tag import Kkma
kkma = Kkma()
from collections import Counter

# load dataset
review = pd.read_csv('./reviewdataset/review_raw.csv')
review[:3] # check

# drop na
review = review[(review['직무']!='\n          전직원\n      ')&(review['직무']!='\n          현직원\n      ')&(review['직무'].notna())]
len(review)

# 장점
review['장점'] = review['장점'].str.replace("\n"," ")
review['장점'] = review['장점'].str.replace("\r"," ")
review['장점'] = review['장점'].str.replace("\t"," ")
review['장점'] = review['장점'].str.replace(",","")

# 단점
review['단점'] = review['단점'].str.replace("\n"," ")
review['단점'] = review['단점'].str.replace("\r"," ")
review['단점'] = review['단점'].str.replace("\t"," ")
review['단점'] = review['단점'].str.replace(",","")

# 요약
review['요약'] = review['요약'].str.replace("\n"," ")
review['요약'] = review['요약'].str.replace("\r"," ")
review['요약'] = review['요약'].str.replace("\t"," ")
review['요약'] = review['요약'].str.replace(",","")

# regular expression
review['장점']= review['장점'].str.replace(pat=r'[^\w]',repl=r' ',regex=True)
review['단점']= review['단점'].str.replace(pat=r'[^\w]',repl=r' ',regex=True)

# new 장단점 column by combining 장점 and 단점 col
review['장단점'] = review['장점']+review['단점']

# extract theme keywords (noun) by 장단점 data
words_noun = []

for rvw in review['장단점']:
    pos = okt.nouns(rvw) # only Nouns
    words_noun.extend(pos)
# print(words_noun)

# # of elements
count_noun = Counter(words_noun)
print(count_noun)

# Assuming count_lists is already created
count_lists = []

# Your existing code to create count lists
for rvw in review['장단점']:
    pos = okt.nouns(rvw)  # only Nouns
    count_noun = Counter(pos)
    count_lists.append(count_noun)

# Convert count lists to a bag-of-words representation
dictionary = corpora.Dictionary(count_lists)
corpus = [dictionary.doc2bow(counts) for counts in count_lists]

# Build the LDA model
lda_model = models.LdaModel(corpus, num_topics=4, id2word=dictionary)

# Print the topics and associated words
for i in range(10):
    print(lda_model.print_topics()[i])
