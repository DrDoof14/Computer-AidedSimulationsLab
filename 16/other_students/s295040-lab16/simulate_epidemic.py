
import random 
import numpy as np 
import hawkesProcess
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from bisect import bisect_left

random.seed(310199)

def simulate_epidemic(t_max, rho_function): 
    t = 0 
    t_i = list()
    t_d = list()
    h_ = dict()
    rates = dict()
    rhos = dict()
    rate = 20 
    t+= random.expovariate(20)
    t_i.append(t)
    if random.random() < 0.02: 
        t_d.append(t)

    while t < t_max: 

        index = bisect_left(t_i, t-20)
        count = len(t_i) - index
        h_sum = 0.05 * count 
        h_[t] = h_sum
        lambda_t = (hawkesProcess.sigma(t) + 2*h_sum)

        if t<20: 
            rate = lambda_t 
            rhos[t] = 0
        else: 
            rate = lambda_t/rho_function(t)
            rhos[t] = rho_function(t)
        
        rates[t] = rate
        t += random.expovariate(rate)
        t_i.append(t)

        if random.random() < 0.02: 
            t_d.append(t)
        
        rates[t] = rate
    rho_squared = lambda t: rho_function(t)**2    
    cost = cost_func(rho_squared)
    return t_i, t_d, cost, h_, rhos, rates

def cost_func(func): 
    integral, error = quad(func, 20, 365)
    return integral 

def sigmoid(t):
    # Set the inflection point of the sigmoid function at t=20
    x = 20-t
    # Set the steepness of the sigmoid function using the parameter "k"
    k = 0.020
    # Set the lower and upper bounds for rho
    rho_min = 0.528
    rho_max = 1
    # Calculate rho using the sigmoid function
    rho = rho_min + (rho_max - rho_min) / (1 + math.exp(-k*x))
    return 1/rho

def constant(t): 
    r = 0.5842
    return 1/r

def stairs(t): 
    if t <= 30: 
        rho = 0.95
    elif t > 30 and t <= 100: 
        rho = 0.75
    elif t > 100 and t <= 150: 
        rho = 0.606
    elif t > 150 and t <= 250: 
        rho = 0.48
    elif t > 250: 
        rho = 0.35

    return 1/rho 

def exp_rho(t):
    value = 0.0041
    rho = math.exp(value*t)
    return rho

def print_rho(func_rho): 
    rhos = dict()
    t = 0 
    while t < 365: 
        if t < 20: 
            rhos[t] = 0
        else: 
            rhos[t] = func_rho(t)
        t+=1
    
    return rhos


def uniform(t): 
        if t < 20: 
            return 1/20 
        else: 
            return 0

def exponential(t): 
    lambda_ = 1/10 
    return lambda_ * math.exp(-lambda_ * t)
