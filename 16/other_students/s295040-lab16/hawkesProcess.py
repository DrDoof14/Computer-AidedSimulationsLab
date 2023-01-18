import random 
import numpy as np 

def sigma(t): 
    if t < 10:
        return 20 
    else: 
        return 0 

def poisson_process(sigma, t_max): 
    #initialize the proecss with N(0)=0
    process=[(0, 0)]
    #initialize the time t 
    t=0
    while t < t_max: 
        tau = random.expovariate(sigma(t))
        n_children = np.random.poisson(2)
        #update the process and time 
        t += tau 
        process.append((t, n_children)) 
    #return the number of ancestor at time t   
    return process 

def uniform(): 
    return random.uniform(0, 20)

def exponential(): 
    lambda_ = 1/10 
    return random.expovariate(lambda_)
    