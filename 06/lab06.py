import numpy as np
from math import *
import matplotlib.pyplot as plt
import math

p=input('For binomial distribution enter 0 and for normal distribution enter 1: ')
if p=='0':
    n = 50
    p = 0.01
    l= list(range(1,n + 1))
    def logbinom(k,n,p):
            temp=(math.log(factorial(n))-(math.log(factorial(n-k)+math.log(k))))+ (k*math.log(p))+((n-k)*math.log(1-p));return math.exp(temp);

        
    dist = [logbinom(k, n, p) for k in l]

    plt.bar(l, dist)
    plt.show()
elif p=='1':
    f=lambda x: np.exp(-(x**2)/2)/(sqrt(2*3.14))

    M=0.3 #scale factor
    M
    u1=np.random.rand(10000)*3  # uniform random samples scaled out
    u2=np.random.rand(10000)    # uniform random samples
    idx,=np.where(u2<=f(u1)/M) # rejection criterion
    v=u1[idx]
    # v
    # print('lenght of v: ')
    # print(len(v))
    fig, ax = plt.subplots()
    ax.set_title('normal distro')
    hist = ax.hist(v, bins=5064)
    plt.show()
    u1_size=u1.size
    v_size=v.size
    efficiency=v_size/u1_size
    print("efficiency factor is equal to {}".format(efficiency))