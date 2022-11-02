import numpy as np
from math import *
import matplotlib.pyplot as plt
import math
from decimal import Decimal
# from decimal import *

distro=input('For binomial distribution enter 1 and for normal distribution enter 2: ')
if distro=='1':
    condition=input("please choose the desired condition:\n 1) n=10  p=0.5\n 2) n=100  p=0.01\n 3) n=1000000   p=0.00001 (time-consuming)\n Your input is: ")
    if condition=="1":
        n=10
        p=0.5
    elif condition=="2":
        n=100
        p=0.01
    elif condition=="3":
        n=1000000  
        p=0.00001
    else:
        print("Please enter the right value.")
        exit()

    l= list(range(1,n + 1))
    def logbinom(k,n,p):
        if condition=='3':
            temp=Decimal(math.log(factorial(n)))-Decimal(math.log(factorial(n-k)))+Decimal((math.log(k)))+ Decimal(k*math.log(p))+Decimal((n-k)*math.log(1-p))
            #to solve he problem of int being too large to convert to float
        else:
            temp=(math.log(factorial(n))-(math.log(factorial(n-k)+math.log(k))))+ (k*math.log(p))+((n-k)*math.log(1-p))

        return math.exp(temp)

    dist = [logbinom(k, n, p) for k in l]

    plt.bar(l, dist)
    plt.show()
elif distro=='2':
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
else:
    print("please enter the right value.")
    exit()