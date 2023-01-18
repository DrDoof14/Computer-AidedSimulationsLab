from math import *
from pympler import asizeof  

import utils 
import fp_utils
import bs_array
import bloom_filters

def main():
    
    # As requested set the number of words in a sentence equal to 6
    s = 6 
    # save the cleaned file
    divina_cleaned = utils.preprocessing(file_path = 'divina_commedia.txt')

    '''
        Sentences stored as list of strings
    '''
    print('\n STORING SENTENCES')
    print('######################################')

    '''
        Create sentences from the file
    '''
    grams, avg_dim = utils.create_senteces(divina_cleaned,s)
    print(f'Number of sentences for 6grams: {len(grams)}')
    print(f'Average size of each sentences in bytes is {avg_dim}')

    '''
        Sentences stored as sets
    '''
    print('######################################')
    print('Data structure: set')
    grams_set = utils.create_senteces_set(grams)
    size = asizeof.asizeof(grams_set)
    print(f'Storage used {size} bytes for 6grams set')

    '''
        Fingerprinting
    '''
    print('\n FINGERPRINTS')
    print('######################################')

    # m is the number of elements on which fingeprinting will be done
    m = len(grams_set)
    # computing Bexp
    print('Computing Bexp, i.e. the minimum value of bits such that no collisions are experienced when storing all the sentences in a fingerprint set')
    bexp = fp_utils.compute_Bexp(grams_set)
    print('Bexp = ',bexp)  
    bexp_fp_set = fp_utils.fingerprint_set(grams_set, n = 2**bexp)

    # computing Bteo
    print('Computing Bteo, i.e. the theoretical number of bit necessary to get a probability of collision x')
    '''here probability of collision is set to 0.5'''
    # given probability of collisions
    prob = 0.5
    bteo = fp_utils.compute_Bteo(m)
    print(f'Bteo for p = {prob} is: ', bteo)

    print(f'Computing the probability of false positve when creating a fingerprint set with numer of bit = Bexp = {bexp}')
    
    false_pos_prob = fp_utils.false_positive_probability(n = 2**bexp, m = m)
    print('False positive probability:', false_pos_prob)

    
    print('\n BIT STRING ARRAY')
    print('######################################')

    # memory considered for fingerprinting
    X = [2**19, 2**20, 2**21, 2**22, 2**23]
    X_str = ["2**19", "2**20", "2**21", "2**22", "2**23"]

    # list containing simulated and theoretical false positive probabilities
    print('\n BIT STRING ARRAY P(FP)')
    emp_probs_list= list()
    th_probs_list= list()
    for x,x_str in zip(X,X_str):
        sim_prob = bs_array.compute_bit_string_array(grams_set,x)
        th_prob = fp_utils.false_positive_probability(x,m)
        print(f'Using x = {x_str}')
        print(f'SIMULATED : False positive probability = {round(sim_prob,5)}')
        print(f'THEORETICAL : False positive probability = {round(th_prob,5)}')
        emp_probs_list.append(sim_prob)
        th_probs_list.append(th_prob)

    '''
        Graph comparing P(FP)
    '''
    bs_array.plot_bs_FP(X,X_str,emp_probs_list,th_probs_list)

    print('\n BLOOM FILTERS')
    print('######################################')
    
    '''
        Computing optimal number of hash functions
    '''
    print('\n BLOOM FILTERS SEARCHING FOR k')
    optimal_ks = list()
    for x,x_str in zip(X,X_str):
        k = bloom_filters.optimal_k(x,m)
        print(f'Using x = {x_str}')
        print(f'Optimal number of hash functions k : {k}')
        optimal_ks.append(k)
    
    '''
        Graph showing optimal number of hash functions k based on the amount of memory dedicated to the fingerprinting
    '''
    bloom_filters.plot_optimal_k(X,X_str,optimal_ks)
    
    '''
        Computing the simulated and theoretical probability of false positive in function of memory, 
        using the optimal number of hash functions 
    '''
    print('\n BLOOM FILTERS P(FP)')
    th_FP_prob_bloom_list = list()
    sim_FP_prob_bloom_list = list()
    for x,x_str,k in zip(X,X_str,optimal_ks):
        th = bloom_filters.FP_prob_bloom(n = x, m = m, k = k)
        emp = bloom_filters.perform_bloom_filter(n=x,k=k,grams_set=grams_set)
        print(f'Using n = {x_str}, optimal k found : {k} \n Theoretical FP prob : {th} , Simulated : {emp}')
        th_FP_prob_bloom_list.append(th)
        sim_FP_prob_bloom_list.append(emp)
    
    bloom_filters.plot_PFP(X,X_str,th = th_FP_prob_bloom_list, sim = sim_FP_prob_bloom_list)

    '''
        For the optional part is set a fixed amount of memory x
    '''
    print('\n OPTIONAL PART')
    print('...please wait...')
    x = 2**19
    bloom_filters.optional_part(k,x,grams_set)

    print('\n ALL DONE!')

if __name__ == '__main__':
    main()