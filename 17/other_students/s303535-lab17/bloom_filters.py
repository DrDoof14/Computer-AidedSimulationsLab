from math import *
import numpy as np
import matplotlib.pyplot as plt

import fp_utils

'''
    Function that compute theoretical false positive probabilty of a bloom filter
    
    Params:
        - n : fingerprinting size
        - m : number of elements to store
        - k : number of hash functions considered
'''
def FP_prob_bloom(n,m,k):
    return pow((1-pow(e,((-k*m)/n))), k) 

'''
    Function used to compute the optimal number of hash functions, 
    given memory (fingerprinting size) n, number of elements to store m
'''
def optimal_k(n,m):
    '''
        Since we don't obtain integer numbers from the formula, 
        we need to check which of the nearest integers have the smallest P(FP)
    '''
    k = (n/m)*log(2)
    k_upper = ceil(k)
    k_lower = floor(k)
    p_lower = FP_prob_bloom(n,m,k_lower)
    p_upper = FP_prob_bloom(n,m,k_upper)
    if p_lower < p_upper:
        return k_lower
    else:
        return k_upper

'''
    Function used to compute bloom filters and relative false positive probability
    FP_prob is computed as the total number of ones in the array / array size, i.e. n
    all elevated to k

    Params:
        - n : fingerprinting size
        - k : number of hash functions considered
'''
def perform_bloom_filter(grams_set,n,k):
    '''
        Since we cannot use k different hash functions, in simulation we modify the input 
        sentence k times. For example if k = 3 and the input sentence is "str", we perform
        fingerprinting of "str", "str"+"1", "str"+"2"
    '''
    # array of zeros
    bloom_filter = np.zeros(n)
    # conflicts counter, in BF a conflict occurs when all the k fingerprints 
    # falls in an already 1 cell
    conflitti = 0
    '''
        For each sentence, first perform fingerprint of the original sentence
        Then we concatenate it with an another string in order to perform bloom filters
    '''
    for el in grams_set:
        #local conflicts counter, if this will be equal to k there will be a "global" conflict
        s = 0
        #compute fingerprint for the original sentence
        fp_0 = fp_utils.compute_fp(el, n)
        #if the array at the position fingerprint is 0 set it to 1 
        #otherwise update local conflicts counter
        if bloom_filter[fp_0] == 0:
            bloom_filter[fp_0] = 1  
        #range if from 1 to k-1, considering fp_0 we have total k fingerprints
        for i in range(1,k):  
            #compute fingerprint for the modified sentence
            fp_k = fp_utils.compute_fp(el+f"{i}",n)
            #if the array at the position fingerprint is 0 set it to 1 
            #otherwise update local conflicts counter
            if bloom_filter[fp_k] == 0:
                bloom_filter[fp_k] = 1

    #compute false positive probability
    prob = (sum(bloom_filter)/n)**k
    
    return prob

'''
    Function in which there is all the optional part
    We want to compare the formula that approximate the number of distinct elements
    stored in a Bf with the reality.
    NOTE: the number of distinct elements stored in the BF is the number of 
    sentences on which we will perform BF, so we will compare it with the output of the formula
    
    Params:
        -k: number of hash functions
        -n: number of bits used in bloom filter
        -grams_set: set of sentences
'''
def optional_part(k,n,grams_set):
    #array of zeros
    bloom_filter = np.zeros(n)
    #list of absolute difference between real and approximated result
    diff = list()
    #list containing the number of sentences considered at each iteration
    counter_list = list()
    #counter of number of sentences considered
    counter_frasi = 0

    sim_list = list()
    '''
        For each sentence, first perform fingerprint of the original sentence
        Then we concatenate it with an another string in order to perform bloom filters
    '''
    for el in grams_set:
        counter_frasi = counter_frasi + 1
        counter_list.append(counter_frasi)
        '''
            here equal to bloom filter
        '''
        fp_0 = fp_utils.compute_fp(el, n)
        if bloom_filter[fp_0] == 0:
            bloom_filter[fp_0] = 1  
        for i in range(1,k):  
            # "modified" sentence
            fp_k = fp_utils.compute_fp(el+f'{i}',n) 
            if bloom_filter[fp_k] == 0:
                bloom_filter[fp_k] = 1
        # N represents the actual number of bits equal to 1 in the bloom filter
        N = np.sum(bloom_filter)
        # compute the given formula
        sim = (-n/k)*log(1-(N/n))
        # append to the list the difference
        diff.append(abs(sim-counter_frasi))

        sim_list.append(sim)
    
    #plt.rcParams["figure.figsize"] = (7,6)
    plt.plot(counter_list,counter_list, color = 'purple', label = 'Actual')
    plt.plot(counter_list,sim_list,color = 'orange', label = 'Estimated')
    plt.grid()
    plt.legend()
    plt.xlabel('Number of sentences considered')
    plt.ylabel('Number of distinct element stored in the BF', fontsize = 8)
    plt.title('OptionalPart: comparison beetween formula and reality')
    plt.savefig('plots/bloom_filters/optional.png')
    plt.show()

    '''
        Plotting difference graph
    '''
    plt.plot(counter_list,diff,color = 'orange')
    plt.grid()
    plt.title('Numerical difference \n of actual and estimated distinct elements stored in the BF')
    plt.xlabel('Number of sentences considered')
    plt.ylabel('Difference')
    plt.savefig('plots/bloom_filters/optional_diff.png')
    plt.show()

'''
    Function to plot the graph showing optimal number of hash functions k 
    based on the amount of memory dedicated to the fingerprinting
'''
def plot_optimal_k(X,X_str,optimal_ks):
    plt.plot(X, optimal_ks, marker = 's', color = 'purple')
    plt.xlabel('Memory (bits)')
    plt.ylabel('Number of hash functions')
    plt.xticks(ticks=X, labels=X_str, fontsize = 7)
    plt.title('Optimal number of HF for the BF')
    plt.grid()
    plt.savefig('plots/bloom_filters/k_bloom.png')
    plt.show()

'''
    Function used to plot the comparison between simulated and theoretical
    false positive probability of a bloom filter in function of 
    the amount of memory dedicated to the fingerprinting
'''
def plot_PFP(X,X_str,th,sim):
    plt.plot(X,sim,marker = 'o', color = 'orange', label = 'Simulated')
    plt.plot(X,th,color = 'purple', label = 'Theoretical')
    plt.xticks(ticks=X, labels=X_str,fontsize = 7)
    plt.xlabel('Memory (bits)')
    plt.title('BloomFilter : false positive probability comparison')
    plt.ylabel('False positive probability')
    plt.grid()
    plt.legend()
    plt.savefig('plots/bloom_filters/PFP_bloom.png')
    plt.show()
