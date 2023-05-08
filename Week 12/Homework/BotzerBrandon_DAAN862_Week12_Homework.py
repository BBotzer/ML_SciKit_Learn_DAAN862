# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:05:34 2022

@author:Brandon Botzer - btb5103



1.Use the following codes to load the assignment12.txt 
    which contains file names. How many file names are in it? (10 points)

file = open("Assignment_12.txt" , 'r')

text1 = file.read()

file.close()



2.Identify the pattern of the file names， and find out how many file names 
match the pattern. (20 points)


3.Find out file names who don't match with the pattern you designed.(20 points)


4.Use the following codes to read the text from 
    “arxiv_annotate1_13_1.txtDownload arxiv_annotate1_13_1.txt” 
in

file = open(“arxiv_annotate1_13_1.txt”, 'r')

text = file.read()

file.close()

Normalize the words and find out their counts.
"""



#imports
import os
import nltk
import re
import pandas as pd

#1.Use the following codes to load the assignment12.txt 
#    which contains file names. How many file names are in it? (10 points)

#set the read path
readpath = "J:\DSDegree\PennState\DAAN_862\Week 12\Homework"
#change the directory
os.chdir(readpath)

file_name = "Assignment_12.txt"

#open the file
file = open(file_name , 'r')
#grab the text as a string
text1 = file.read()
#close the file
file.close()

#use regex to split text1 by any number of blank spaces
file_list = re.split('\s+', text1)

#number of filenames in text1
num_filenames = len(file_list)

print("\nSplitting the files by spaces, there are " + str(num_filenames) + 
      " files in " + file_name)



#2.Identify the pattern of the file names， and find out how many file names 
#match the pattern. (20 points)

#The commented section here works as well as the function below
"""
#decleare the pattern
pattern = r'[a-zA-Z]+_[a-zA-Z]+[0-9]+_[0-9]+_[0-9]+\.[a-zA-Z]+'
#build the regex object with the pattern
regex = re.compile(pattern, flags = re.IGNORECASE)
#Get the list of cases that match the pattern
reg_list = regex.findall(text1)
#find the length of the list of  matching cases
reg_len = len(reg_list)

print("Using Regex, there are " + str(reg_len) + " files in " + file_name)
print("This was due to typos in some of the file names.")
"""

#I tried to use the regex-generator for the pattern 
#but it would not work correctly.  I used it as a start and then 
#modified it.

#Pattern from regex-generator shown here:
    #r"^[a-zA-Z]+_[a-zA-Z]+([0-9]+(_[0-9]+)+)\.[a-zA-Z]+$"

def use_regex(input_text):
    #compile the regex object
    regex = re.compile(r"[a-zA-Z]+_[a-zA-Z]+[0-9]+_[0-9]+_[0-9]+\.[a-zA-Z]+", 
                       re.IGNORECASE)
    #find all cases that match the pattern and return them
    return regex.findall(input_text)

#Get the list of cases that match the pattern
reg_list = use_regex(text1)
#find the length of the list of  matching cases
reg_len = len(reg_list)

print("\nUsing Regex, there are " + str(reg_len) + " files in " + file_name)
print("This was due to typos in some of the file names.")




#3.Find out file names who don't match with the pattern you designed.(20 points)

#Use sets to effectivly look for the missing pieces

#Bring the lists into sets
set_a = set(file_list)
set_b = set(reg_list)

#Full set - partial set = missing files  (similar to an outer join)
#put it back into a list so all of my lists of files are var type list
typo_list = list(set_a - set_b)

print("\nThe files which were spelled incorrectly are: \n")
print(typo_list)




#4.Use the following codes to read the text from 
#    “arxiv_annotate1_13_1.txt"

#open and read in the file
file_name2 = "arxiv_annotate1_13_1.txt"
file = open(file_name2, 'r')
text2 = file.read()
file.close()


#Normalize the words and find out their counts.


#First we'll remove special characters
#Taken from "Text Analytics with Python, 2nd ed."
def remove_special_characters(text, remove_digits = False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

text2 = remove_special_characters(text2)



#break the large string up into a list of words
#use regex to split text1 by any number of blank spaces
words = re.split('\s+', text2)
#Try a nltk word tokenizer for possible different results
words_token = nltk.word_tokenize(text2)
#Try a split() function for possible differenet results
words_split = text2.split(' ')

#I will go with the word_tokenizer as it removed the empty string


#find some word counts
#number of words in the list
num_words = len(words_token)
#unique words
unique_words = list(set(words_token))
#number of unique words
num_unique_words = len(unique_words)

print("\nFrom the Arxiv file, there are by the NLTK word_tokenizer:")
print("  " + str(num_words) + " words.")
print("  " + str(num_unique_words) + " unique words.\n")



#we can also do this with a FreqDist
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


#find the unique words from the word list
dist = FreqDist(words_token)
print("The FreqDist function likewise reports " 
      + str(len(dist)) + " unique words.")


print("\n\nSome of these words may come from the same root stem/word.")
print("We'll look at both root stems and root words here.\n")


#Root Stemming
#build the Porter Stemmer object
porter = nltk.PorterStemmer()
#Get the root stems for each word
word_stems = [porter.stem(word) for word in words_token]

#Get the distribution of the stem'd words
stem_dist = FreqDist(word_stems)


#Download for functionallity if needed
nltk.download('omw-1.4')

#Lemmatization
#Build the WordNetLemmatizer object
WNLemma = nltk.WordNetLemmatizer()
#Get the Lemmatization of the words
word_lemma = [WNLemma.lemmatize(word) for word in words_token]
#lemma distribution
lemma_dist = FreqDist(word_lemma)


#display the counts
#make a data frame of all of the different word grabs and counts

#dist
#stem_dist
#lemma_dist

#Get tuples of the distributions
tup_words = tuple(dist.items())
tup_stem = tuple(stem_dist.items())
tup_lemma = tuple(lemma_dist.items())

#create data frames and join them together...
#there is probably a better way to do this using a pivot but here we are.

twdf = pd.DataFrame(data = tup_words, columns = ["words", "word_count"])
tsdf = pd.DataFrame(data = tup_stem, columns = ["stems", "stem_count"])

#join the tuple stem data frame to the tuple word data frame on the left
#this keeps all values and will just join the frames together
df = twdf.join(tsdf, how='left', sort = True)

tldf = pd.DataFrame(data = tup_lemma, columns = ["lemmas", "lemma_count"])

#join the lemma data frame to the word/stem data frame on the left 
#this keeps all values and will just join the frames together
df = df.join(tldf, how='left', sort = True)

print("\nThe word, stem, and lemma distribution counts:\n")
print("You'll notice that eventually the stem and lemma columns will " + 
      "not match up with the words column as the tokenizers interpret " +
      "differently.  In the case of lemma, it missed the word 'a'.  " + 
      "It also had trouble with strange breakes such as 'eg.' which " +
      "in the text file was presented as 'e g' and thus taken as two words.\n")

print("\nI attempted to make use of Dipanjan's sequence of text to compare " +
      "normalization results but there seems to be an issue with a " +
      "dependancy I was unable to resolve." )

#print all of the words, counts, stems, stem_counts, lemmas, and lemma_counts
print(df.to_string())









"""

#Here I now bring in Dipanjan's sequence of text normalization
#IT is having issues with spaCy and using fileno for file imports

#Strip HTML
import re
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


#Removing Accented Characters
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#Expanding Contractions
from contractions import contractions_dict
import re

def expand_contractions(text, contraction_mapping=contractions_dict):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#Remvoing Special Characters
#You can find this above

#Case Conversion
#Use the .lower()

#Text Correction:
    
    #Correcting Repeating Characters
from nltk.corpus import wordnet
def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens




    #Correcting Spellings
import re, collections

#make the vocabulary
def tokens(text): 
    """
    #Get all words from the corpus
"""
    return re.findall('[a-z]+', text.lower()) 

WORDS = words
WORD_COUNTS = collections.Counter(WORDS)

#Define set of words that are one, two, or threee away from out input
def edits0(word): 
    """
    #Return all strings that are zero edits away 
    #from the input word (i.e., the word itself).
"""
    return {word}



def edits1(word):
    """
    #Return all strings that are one edit away 
    #from the input word.
"""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        """
        #Return a list of all possible (first, rest) pairs 
        #that the input word is made of.
"""
        return [(word[:i], word[i:]) 
                for i in range(len(word)+1)]
                
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    #Return all strings that are two edits away 
    #from the input word.
"""
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

#Define function to return a subset of words from our candidate set of words
#obtained from the edit functions based on if they occur in the vocabuary
#of the tokenfunction
def known(words):
    """
    #Return the subset of words that are actually 
    #in our WORD_COUNTS dictionary.
"""
    return {w for w in words if w in WORD_COUNTS}

#Select the correct word from a number of candidates
def correct(word):
    """
    #Get the best correct spelling for the input word
"""
    # Priority is for edit distance 0, then 1, then 2
    # else defaults to the input word itself.
    candidates = (known(edits0(word)) or 
                  known(edits1(word)) or 
                  known(edits2(word)) or 
                  [word])
    return max(candidates, key=WORD_COUNTS.get)

#We now go back and check case and ensure a correct match
#and show the correct text
def correct_match(match):
    """
    #Spell-correct word in match, 
    #and preserve proper upper/lower/title case.
"""
    
    word = match.group()
    def case_of(text):
        """
        #Return the case-function appropriate 
        #for text: upper, lower, title, or just str.:
"""
        return (str.upper if text.isupper() else
                str.lower if text.islower() else
                str.title if text.istitle() else
                str)
    return case_of(word)(correct(word.lower()))

    
def correct_text_generic(text):
    """
    #Correct all the words within a text, 
    #returning the corrected text.
"""
    return re.sub('[a-zA-Z]+', correct_match, text)



    #word correction for spelling can also be done with the textblob library
#from textblob import Word
#w = Word('flaot')
#w.spellcheck()


#Stemming Words
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


#Lemmatization The book's version is having issues so I'll make my own
#import spacy
# use spacy.load('en') if you have downloaded the language model en directly after install spacy
#nlp = spacy.load('en_core', parse=True, tag=True, entity=True)
#text = 'My system keeps crashing his crashed yesterday, ours crashes daily'

#def lemmatize_text(text):
#    text = nlp(text)
#    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
#    return text


def lemmatize_text_botzer(text):
    WNLemma = nltk.WordNetLemmatizer()
    
    #tokenize text
    words_token = nltk.word_tokenize(text)
    #lemmatize
    word_lemma = [WNLemma.lemmatize(word) for word in words_token]
    
    text = ' '.join(word_lemma)
    return text
    


#Remmove Stopwords
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



#All of the Dipanjan funcitons build into one normalizer
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_stemmer=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # Stem text
        if text_stemmer:
            doc = simple_stemmer(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus




#and this doesn't work as intended...
test = normalize_corpus(text2)


"""


















