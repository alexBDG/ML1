import numpy as np
import os


###############################################################################
# find the most frequent words of a data set
###############################################################################


def find_frequent_words(data,n):
    words = {} 
    hf_words = {} 
    for data_point in data:
        for word in data_point.get('text').split(' '):
            if(word.lower()) in words:
                words[word.lower()] += 1
            else:
                words[word.lower()] = 1
      
    words.pop('')
  
    for i in range(0,n):
        maxi = 0
        word_maxi = ""
        for word in words:
            freq = words.get(word)
            if(freq>maxi):
                maxi = freq
                word_maxi = word
        hf_words[word_maxi] = maxi
        words.pop(word_maxi)
        
    return(hf_words)
    
    
###############################################################################
# save the most frequent words
###############################################################################
  

def savewords(data,n):
    hf_words = find_frequent_words(data,n)
    with open(os.getcwd()+"/words.txt","w") as file:
        for hfw in hf_words:
            file.write("{0} : {1}\n".format(hfw,hf_words['{0}'.format(hfw)]))


###############################################################################
# bluid our matrix of features
###############################################################################


def build_system(data,istart,iend,nbw,hf_words):
    nb = iend-istart
    A = np.zeros((nb,4+nbw))
    b = np.zeros(nb)
    for i in range(0,nb):
        A[i,0] = 1.
        A[i,1] = float(data[i+istart].get('children'))
        A[i,2] = float(data[i+istart].get('is_root'))
        A[i,3] = float(data[i+istart].get('controversiality'))
        k=1    
        for hfw in hf_words:
            for word in data[i+istart].get('text').split(' '):
                if (hfw == word.lower()):
                    A[i,3+k] += 1
            k+=1
        b[i] = float(data[i+istart].get('popularity_score'))
  
    AA = np.dot(np.transpose(A),A)
    Ab = np.dot(np.transpose(A),b)
    return(A,b,AA,Ab)
    
  
###############################################################################
# bluid our matrix of features + 2 
###############################################################################

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def cleaning(text):
    words = word_tokenize(text)
    tokens = [w.lower() for w in words if w.isalpha()]
    stopw = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopw]
    return tokens

def build_system2(data,istart,iend,nbw,hf_words):
    nb = iend-istart
    A = np.zeros((nb,4+nbw+2))
    b = np.zeros(nb)
    for i in range(0,nb):
        A[i,0] = 1.
        A[i,1] = float(data[i+istart].get('children'))
        A[i,2] = float(data[i+istart].get('is_root'))
        A[i,3] = float(data[i+istart].get('controversiality'))
        k=1    
        for hfw in hf_words:
            for word in data[i+istart].get('text').split(' '):
                if (hfw == word.lower()):
                    A[i,3+k] += 1
            k+=1
        ########
        word = cleaning(data[i+istart].get('text'))
        A[i,4+nbw] = len(word)
        ########
        A[i,4+nbw+1] = (float(data[i+istart].get('is_root')) + 3000 ) * len(word)
        ########
        b[i] = float(data[i+istart].get('popularity_score'))
  
    AA = np.dot(np.transpose(A),A)
    Ab = np.dot(np.transpose(A),b)
    return(A,b,AA,Ab)

