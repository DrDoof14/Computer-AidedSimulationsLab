import re 
from math import *
from pympler import asizeof  

'''
    Function used to preprocess the textual input
'''
def preprocessing(file_path = 'divina_commedia.txt'):
    #read the file containing divina commedia
    divina_commedia = open(file_path,'r').read().lower()
    #remove punctuaction
    divina_commedia = re.sub(r'[^\w\s]', '', divina_commedia)
    #split in verses
    divina_commedia = divina_commedia.split("\n")
    #list that will be used to store the final output of the text preprocessing
    divina_cleaned = list()

    '''remove titles'''
    for versi in divina_commedia:
        if versi == '' or versi.startswith("inferno canto")  or versi.startswith("purgatorio canto")  or versi.startswith("paradiso canto"):
            divina_commedia.remove(versi)
    '''create list of words'''
    for versi in divina_commedia:
        divina_cleaned.extend(versi.split(' '))
    '''removing empty string words'''
    for word in divina_cleaned:
        if word == '':
            divina_cleaned.remove(word)
    
    return divina_cleaned

'''
    Function used to create sentences, of fixed lenght s, from a bag of word
    It also compute average size of each sentence in bytes
'''
def create_senteces(bow, s):
    grams = [" ".join(bow[i:i+s]) for i in range(len(bow)-s+1)]
    sum = 0
    for sentence in grams:
        dim_bytes = asizeof.asizeof(sentence)
        sum = sum + dim_bytes
    avg_dim = round(sum / len(grams),2)
    return grams, avg_dim

'''
    Given a list of strings it stores all the strings in a python set
'''
def create_senteces_set(_grams):
    grams_set = set()
    for sent in _grams:
        grams_set.add(sent)
    return grams_set


