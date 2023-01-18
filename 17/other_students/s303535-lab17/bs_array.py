import numpy as np
import matplotlib.pyplot as plt

import fp_utils

'''
    Function used to compute bit string array and relative false positive probability
    FP_prob is computed as the total number of ones in the array / array size, i.e. n

    Params:
        -grams_set: set of sentences
        -n : fingerprinting size
'''
def compute_bit_string_array(grams_set, n):
    #create an array of zeros of lenght n
    bit_string_array = np.zeros(n)
    #conflicts counter
    conflitti = 0
    for el in grams_set:
        #for each sentence, compute fingerprint
        fp = fp_utils.compute_fp(el, n)
        #if the array at the position fingerprint is 0 set it to 1 
        #otherwise update conflicts counter
        if bit_string_array[fp] == 0:
            bit_string_array[fp] = 1

    #compute false positive probability 
    prob = sum(bit_string_array)/n

    return prob

'''
    Function used to plot the comparison graph of theoretical and simulated 
    false positive probability
'''
def plot_bs_FP(X,X_str, sim, th):

    '''Graph showing COMPARISON between simulated and theoretical FP prob'''
    plt.plot(X, sim, marker = 'o', label = 'Simulated', color = 'orange')
    plt.plot(X, th,label = 'Theoretical', color = 'purple')
    plt.xlabel('Memory (bits)')
    plt.ylabel('False positive probability')
    plt.xticks(ticks=X, labels=X_str, fontsize=7 )
    plt.legend()
    plt.title('BitStringArray: false positive probability comparison')
    plt.grid()
    plt.savefig('plots/bit_string_array/PFP_bit_string.png')
    plt.show()