import hashlib 
from math import *
import numpy as np

'''
    Function used to compute fingerprinting

    Params:
        -element: element on which fingerprinting is perfomed
        -n : fingerprinting size
'''
def compute_fp(element,n):
    # md5 hash
    element_hash = hashlib.md5(element.encode('utf-8')) 
    # md5 hash in integer format
    element_hash_int = int(element_hash.hexdigest(), 16) 
    # map into [0,n-1]
    h = element_hash_int % n 

    return h

'''
    Given a set, it computes tha fingerprint for each element 
    and store this fingerprint in another set

    Params:
        -grams_set: set of sentences
        -n : fingerprinting size
'''    
def fingerprint_set(grams_set,n):
    grams_fp_set = set()

    for element in grams_set:
        h = compute_fp(element,n)
        grams_fp_set.add(h)

    return grams_fp_set

'''
    Function used to compute Bexp, i.e. the minimum value of bits such that no 
    collisions are experienced when storing all the sentences in a fingerprint set.
    Precisely it finds the best exponent of 2**exp, where exp means that we store 
    an elemen as a exp-bit value.
    The function starts from bit = 1, so from 2**1, and it iteratively increases the exponent

'''
def compute_Bexp(gram_set, bit = 1):
    ''' 
        This function is recursively modelled
        Base case : if I find a conflict
        Inductive case : if I don't find a conflict
    '''
    fp_list = list()
    for el in gram_set:
        fp_list.append(compute_fp(el,n=2**bit))
    '''
        Check the len since set does not accept duplicates;
        so if they have different lenghts  at least one conflict has occured
    '''
    if len(fp_list) != len(set(fp_list)):
        return 0 + compute_Bexp(gram_set, bit = bit + 1)
    else: 
        return bit

'''
    Function used to compute Bteo, i.e. the theoretical number of bit 
    necessary to get a probability of collision x, see report for further explanations

    Params:
        -m: number of element needed to be stored
        -prob: probability of conflicts we want to obtain

'''
def compute_Bteo(m):
    return np.floor(np.log2((m/1.17)**2))

'''
    Compute the theoretical probability of having false positives when fingerprinting
    
    Params:
        -n: fingerprinting size
        -m: number of elements needed to be stored
'''
def false_positive_probability(n,m):
    return 1 - pow((1-1/n),m)


