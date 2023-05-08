# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:18:20 2018

@author: Leo
"""

# text mining

import re


# Regular expression

text2 = 'Python is   \t        a   great      tool for\t data analysis'
re.split('\s+', text2)      # \s+ means more than one space.

regex = re.compile('\s+')   # compile the pattern
regex.split(text2)          # split text2 by the pattern
regex.findall(text2)        # find all patterns in text2


text3 = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""

pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)  # compile a pattern for emails
m = regex.match('wesm@bright.net, edd')     # check if a text match the pattern
m.start()          #  the first index of the text matching with pattern
m.end()            # the last index of the text matching with pattern
'wesm@bright.net, edd'[m.start(): m.end()] # email in the text


text3 = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
regex.findall(text3)
m = regex.search(text3)
m
text3[m.start():m.end()]



##############################################################
import nltk
# Tokenization
text2 = '''This is the first sentense. Nothing happened.
 Is this the third one? Yes, it is'''
nltk.sent_tokenize(text2)


text1 = '''By "natural language" we mean a language that is used 
for everyday communication by humans'''
text1.split(' ')
nltk.word_tokenize(text1)



from nltk.book import *
text2
sent2

# counting vocabulary of words
len(text2)                  # How many words in text2
len(set(text2))             # How many unique words in text2
list(set(text2))[:10]       # Show 10 unique words
len(sent2)                  # How many words in sent2

# Frequency of words
dist = FreqDist(text2)     # Return a dictionary with words as keys and their freq as values
len(dist)                  # How many unique words
words = dist.keys()        # Unique words
list(words)[:10]           # Show head of words

# find all words which contains more than 5 chars 
# and whose frequency larger than 100
freqwords = [w for w in words if len(w) > 5 and dist[w] > 100]  
freqwords[:10]

# NOrmalization and stemming
example1 = 'reply Replying replied REPLIES' 
words1  = example1.lower().split(' ')
words1

porter = nltk.PorterStemmer()
[porter.stem(w) for w in words1]

#Lemmatization
sent3
[porter.stem(w) for w in sent3]

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(w) for w in sent3]


