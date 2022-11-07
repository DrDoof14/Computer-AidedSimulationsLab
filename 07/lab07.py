import numpy as np
import pandas as pd
from scipy.stats import uniform # to create uniform continuous random variables
import math
from datetime import datetime
import time 
from scipy.stats import t # to create Student’s t continuous random variables
import matplotlib.pyplot as plt 
import seaborn as sns





def random_dropping (n, trials,d):
    #d = 0 -> random

    
    if d == 0:
        results = np.zeros(trials, ) #to have empty bin as the number of 
        for r in range(trials):
            bins = np.zeros(n,)
            for b in range(n):
                index = np.random.randint(low = 0, high = n, size = 1)
                bins[index] = bins[index] + 1
                results[r] = max(bins)

        avg = np.average(results)
        std = np.std(results, ddof=1)
        if std!=0:
            confInt = t.interval(0.95, trials-1, avg, std/math.sqrt(trials))

        else:
            confInt = (avg,avg)

        return avg, confInt
    
    elif d == 1  or d == 2 :
        results = np.zeros(trials, )
        for r in range(trials):
            bins = np.zeros(n,)
            for b in range(n):
                indexes = np.random.randint(low = 0, high = n, size = d)
                minB = bins[indexes[0]]
                minIdx = indexes[0]
                for i in range(d):
                    if minB > bins[indexes[i]]:
                        minIdx = indexes[i]
                        minB = bins[indexes[i]]
                bins[minIdx] = bins[minIdx] + 1
            results[r] = max(bins)

        avg = np.average(results)
        std = np.std(results, ddof=1)
        if std!=0:
            confInt = t.interval(0.95, trials-1, avg, std/math.sqrt(trials))
        else:
            confInt = (avg,avg)
        return avg, confInt




#we perform all the simulations since we need to compare them in a plot


#Input parameters
bins = [100, 200, 400, 600, 800, 1000] # number of bins 
trials = 20 #the number of times the simulations happens 
seed = 1886


#Simulations for Random Dropping
avgs = []
confInts = []
np.random.seed(seed) 
for b in bins:
    avg, confInt= random_dropping(b, trials,0)
    avgs.append(avg)
    confInts.append(confInt)
#Output parameters
avgs = np.array(avgs)
confInts = np.array(confInts)


#Simulations for d = 2
avgs2 = []
confInts2 = []
np.random.seed(seed)
for b in bins:
    avg, confInt = random_dropping(b, trials,1 )
    avgs2.append(avg)
    confInts2.append(confInt)
#Output parameters
avgs2 = np.array(avgs2)
confInts2 = np.array(confInts2)



#Simulations for Random Load Balancing with d = 4
avgs4 = []
confInts4 = []
np.random.seed(seed)
for b in bins:
    avg, confInt = random_dropping(b, trials,2 )
    avgs4.append(avg)
    confInts4.append(confInt)
#Output parameters
avgs4 = np.array(avgs4)
confInts4 = np.array(confInts4)





#visualization for random dropping
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(np.log10(bins), avgs, color = 'b', marker = 'o', label = 'Random Dropping')
ax.fill_between(np.log10(bins), confInts[:,0], confInts[:,1],alpha = 0.2, label = 'Random Dropping 95% Confidence Interval')
ax.legend()
plt.xlabel('log10(Bins)')
plt.title('Simulation results for randomized dropping policies')
plt.ylabel('Max bin occupancy')
plt.show()


#visualization for d=2

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(np.log10(bins), avgs2, color = 'orange', marker = 'x', label = 'Random Load Balancing d = 2')
ax.fill_between(np.log10(bins), confInts2[:,0], confInts2[:,1],alpha = 0.2, label = 'Random Load Balancing d = 2 95% Confidence Interval')
ax.legend()
plt.xlabel('log10(Bins)')
plt.title('Simulation results for randomized dropping policies')
plt.ylabel('Max bin occupancy')
plt.show()


#visualization for d=4

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(np.log10(bins), avgs4, color = 'g', marker = '^', label = 'Random Load Balancing d = 4')
ax.fill_between(np.log10(bins), confInts4[:,0], confInts4[:,1],alpha = 0.2, label = 'Random Load Balancing d = 4 95% Confidence Interval')
ax.legend()
plt.xlabel('log10(Bins)')
plt.title('Simulation results for randomized dropping policies')
plt.ylabel('Max bin occupancy')
plt.show()
