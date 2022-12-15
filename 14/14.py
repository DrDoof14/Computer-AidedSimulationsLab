import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
#########################################
lambda_values=[0.6, 0.8, 0.9,0.95, 0.99, 1.01, 1.05, 1.1, 1.3]
np.random.seed(32)
#########################################
generation_limit=30
runs=1000
##########################################
#we should store all the generation that have survived untill the generation that is extinct 
# since technically we're storing the survived generations 
final_dict_lambda=defaultdict(list) # dictionary for survival 
final_extinct_dict=defaultdict(list) # dictionary for extinction 
for lam in lambda_values:
    gen_dict_survive={}
    gen_dict_extinct={}
    for i in range(runs):
        generation_dict=defaultdict(list)
        generation = 0
        generation_dict[generation].append(np.random.poisson(lam)) # first person
        #conditions in which the simulation stops and goes to another one:
        # 1-there are no more children in a generation and 2- which is the limit we've put ourselves 
        #it goes till a specific generation and then stops 
        while (sum(generation_dict[generation])!=0) and  (generation<generation_limit):
            temp=generation+1  #since we have to create the children in the next generation
            for _ in range(sum(generation_dict[generation])):
                generation_dict[temp].append(np.random.poisson(lam))#creating the random numbers for each r.v.(Y)
            generation= generation+1 # go to the next generation 
        for j in range(generation): # for each iteration, we add increment the generations that survived
            # for example if extinction has happened in generation 10 , we increment the value for generations 
            # 0 to 9 
            if j in gen_dict_survive.keys():
                gen_dict_survive[j]+=1 
                gen_dict_extinct[j]-=1
            else:
                gen_dict_survive[j]=1
                gen_dict_extinct[j]=runs-1 # the number of extinctions is (the number of runs - number of survivals)
    for c , m in gen_dict_survive.items():
        gen_dict_survive[c]= m /runs # finding the probability 
    final_dict_lambda[lam]=gen_dict_survive  # to store the number of survivals 
    for k , v in gen_dict_extinct.items():
        gen_dict_extinct[k]= v /runs # finding the probability 
    final_extinct_dict[lam]=gen_dict_extinct # to sotre the number of extinctions

###################################################################
for k,v in final_dict_lambda.items():
#     plt.figure(figsize=(10,10))
    plt.plot(v.keys(),v.values(),label=f'{k}')
    plt.xlabel('Generations')
    plt.ylabel('Probability of Survival')
    plt.legend()
    plt.grid()
plt.show()
    #####################################################
for k,v in final_extinct_dict.items():
#     plt.figure(figsize=(10,10))
    plt.plot(v.keys(),v.values(),label=f'{k}')
    plt.xlabel('Generations')
    plt.ylabel('Probability of Extinction')
    plt.grid()
    plt.legend()
plt.show()