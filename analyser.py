import nltk
import re
import string
import gensim
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from lda import lda


def countWords(string):
    return len(string)

def uniqueWords(string):
    return list(Counter(string))

def countUnique(string):
    return len(Counter(string)) 

def nonStopWords(string):
    stop_words = set(stopwords.words('english'))
   # words = word_tokenize(" ".join(string))
    nonstopText = []
    for word in string:
        if word not in stop_words:
            nonstopText.append(word)
    return nonstopText

def nonStopCount(string):
    return countWords(nonStopWords(string))

def averageWordLength(string):
    return sum(len(word) for word in string) / len(string)

def sentiment_textblob(feedback): 
    senti = TextBlob(feedback) 
    polarity = senti.sentiment.polarity    
    return polarity

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

from rake_nltk import Rake
def num_keywords(title):
    meta = str('BY SETH FIEGERMAN JAN 07, 2013')
    metaRes = [word.strip(string.punctuation) for word in meta.split()]
    #print(countWords(metaRes))
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(title)
    #print(r.get_ranked_phrases()) # To get keyword phrases ranked highest to lowest.
    result = len(r.get_ranked_phrases())  + countWords(metaRes) -1
    return result

#print(TextBlob(' '.join(title)).subjectivity)

def PosNeuNeg(string):
    sid = SentimentIntensityAnalyzer()
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]
    for word in string:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)    
    return (pos_word_list,neu_word_list,neg_word_list) 

def subjectivity(string):
    return TextBlob(string).subjectivity

def abs_title_subjectivity(subject):
    value = float(subject)
    if value == 0:
        value = 0.5
        return value
    if value >=  0.5:
        value = value - 0.5
        return value
    if value > 0 and value < 0.5:
        value = 0.5 - value
        return value
    
file = open('content.txt', 'r')
read_file = file.read()

file1 = open('title.txt', 'r')
read_file1 = file1.read()

file_content = str(read_file)
content = [word.strip(string.punctuation) for word in file_content.split()]
while("" in content) : 
    content.remove("")

file_title = str(read_file1)
title = [word.strip(string.punctuation) for word in file_title.split()]
while("" in title) : 
    title.remove("")

numLinks = 3 #Number of Links
numVideos = 0 #Number of Videos
numImages = 1 #Number of Images

countwordsT = countWords(title) #Number of Words Title

countwordsC = countWords(content) #Number of Words Content
##
countunique = countUnique(content) #Number of Unique Words
#print(countunique)
#
#print(nonStopWords(content))
nonstopCount = nonStopCount(content) # Number of non-Stop Words
#print(nonstopCount)

rateNonStopWords = 0.999999995192
# Rate of non-Stop Words
#
rateUniqueNonStopWords = nonStopCount(uniqueWords(content))/nonstopCount # Rate of Unique non-Stop Words

average_token_length = averageWordLength(content) # Average Words Length

global_subjectivity = TextBlob(' '.join(content)).subjectivity
title_subjectivity = TextBlob(' '.join(title)).subjectivity
global_sentiment_polarity = TextBlob(' '.join(content)).polarity 
title_sentiment_polarity = TextBlob(' '.join(title)).polarity
LDA = lda('content.txt')

print('n_tokens_title =',countwordsT)
print('n_tokens_content =',countwordsC)

print('n_unique_tokens =',countunique)
print('n_non_stop_words =',rateNonStopWords)
print('n_non_stop_unique_tokens =',rateUniqueNonStopWords)

print('num_href =',numLinks)
print('num_imgs =',numImages)
print('num_videos =',numVideos)
print('average_token_length =',average_token_length)
num_keywords=num_keywords(" ".join(title))
print('num_keywords =', num_keywords)

is_workday = 1
is_weekend = 0
print('is_workday=', is_workday)
print('is_weekend=', is_weekend)


print('LDA00 =',LDA[0][1])
print('LDA01 =',LDA[1][1])
print('LDA02 =',LDA[2][1])
print('LDA03 =',LDA[3][1])
print('LDA04 =',LDA[4][1])

print('global_subjectivity =',global_subjectivity)
print('global_sentiment_polarity=', global_sentiment_polarity)

avg_positive_polarity = 0.35
min_positive_polarity = 0.1
max_positive_polarity = 0.75
abs_title_subjectivity= abs_title_subjectivity(title_subjectivity)
abs_title_sentiment_polarity = abs(title_sentiment_polarity)
print('avg_positive_polarity=',avg_positive_polarity)
print('min_positive_polarity=',min_positive_polarity)
print('max_positive_polarity=',max_positive_polarity)
print('title_subjectivity=',title_subjectivity)
print('title_sentiment_polarity=',title_sentiment_polarity)
print('abs_title_subjectivity=',abs_title_subjectivity)
print('abs_title_sentiment_polarity=',abs_title_sentiment_polarity)

result = str(countwordsT) + ' ' + str(countwordsC) + ' ' + str(countunique) + ' ' + str(rateNonStopWords) + ' ' + str(rateUniqueNonStopWords) + ' ' + str(numLinks) + ' ' + str(numVideos) + ' ' + str(numImages) + ' ' + str(average_token_length) + ' ' + str(num_keywords) + ' ' + str(is_workday) + ' ' + str(is_weekend)+ ' ' + str(LDA[0][1]) + ' ' + str(LDA[1][1]) + ' ' + str(LDA[2][1]) + ' ' + str(LDA[3][1]) + ' ' + str(LDA[4][1]) + ' ' + str(global_subjectivity) + ' ' + str(global_sentiment_polarity) + ' ' + str(avg_positive_polarity) + ' ' + str(min_positive_polarity) + ' ' + str(max_positive_polarity) + ' ' + str(title_subjectivity) +' ' + str(title_sentiment_polarity) + ' ' + str(abs_title_subjectivity) + ' ' + str(abs_title_sentiment_polarity) 
f = open("result.txt", "w")
f.write(result)
f.close()