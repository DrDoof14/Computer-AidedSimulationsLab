#importing the libraries
import hashlib
import re
from pympler import asizeof
import math
import matplotlib.pyplot as plt
import numpy as np
from bitarray import bitarray 
# Pre-processing phase
# ==============================
#importing the .txt file 
in1=input('Press Enter to start the simulation\n The first part is the Pre-processing phase\n')
file = open("commedia.txt", "rt")
data = file.read() 
#===========================
data=re.sub(r"[^\w\d'\s]+",'',data) # deleting the punctuations
words = data.split('\n')
del words[0:8]
clean_words=[]
for line in words.copy():
    if line.startswith("Inferno") or line.startswith("Purgatorio") or line.startswith("Paradiso")or not line.strip():
        words.remove(line) # deleting the headers 
    else:
        line = line.strip()
        splited_words = line.split(" ")
        clean_words.extend(splited_words)
print('the total number of words is equal to\n')
print(len(clean_words)) #counting the number of words 
print(f'The size of distinct words is equal to {asizeof.asizeof(clean_words)} bytes \n')
print(f'number of distinct words is equal to {len(set(clean_words))} \n')
#=====================================================
#creating the verses 

plagiarism_size= 6 #window size 
verses = []
finger_prints=[]
verse_count=0



for i in range (len(clean_words)):
    if len(clean_words[i:i+plagiarism_size])==plagiarism_size:
        verses.append(' '.join(clean_words[i:i+plagiarism_size]))

verses_set=set(verses)    
m = len(verses_set)
epsilon = 10 ** -5
b = math.ceil(math.log((m/epsilon) , 2 )) # b has to be larger or equal to this value 
n = 2 ** b


for i in verses_set:
    temp=i
    encoded=temp.encode('utf-8')
    hashed=hashlib.md5(encoded)
    int_hashed=int(hashed.hexdigest(), 16)
    h = int_hashed % n
    finger_prints.append(h)


        
finger_prints_set=set(finger_prints)       
#we should create the fingerprints using a hashfunction 
#then we should compute and store the fingerprints inside of a set 
print(f'\nTotal number of verses is {len(verses)} \n')
print(f'\nTotal number of verses(set) is {len(verses_set)} \n')
print(f'\nTotal number of fingerprints is {len(finger_prints)} \n')
print(f'\nTotal number of fingerprints(set) is {len(finger_prints_set)} \n')
print (f'The number of collisions is equal to {len(finger_prints)-len(finger_prints_set)} \n')
in0=input('Press enter to see the size of the set of sentences\n')
#====================Set of sentences=======================================================
print(f'The size of the set of senteces is equal to {asizeof.asizeof(verses_set)/1024} KB\n') # 
prob_false_positive_set_of_senteces=(m/n)
# epsilon = m/n = m/ 2**b
in444=input('do you want to see the probability of false positive for the set of sentences? Y/N \n')
if in444=='y'or in444=='Y' or in444=="yes" or in444=='Yes':
    print(prob_false_positive_set_of_senteces)
#=========================================================

#it's better to have both fingerprints and the verses stored in a dictionary 
# so we're gonna do it from the beginning
new_verses=[]
# m = len(verses) 
# we already have the value for m , b , epsilon and n e
# epsilon = 10 ** -4
# b = math.log((m/epsilon) , 2 ) # b has to be larger or equal to this value 
# n = m * b
general_dict={}

for i in range (len(clean_words)):
    if len(clean_words[i:i+plagiarism_size])==plagiarism_size:
        new_verses.append(' '.join(clean_words[i:i+plagiarism_size]))
        temp=new_verses[i]
        encoded=temp.encode('utf-8')
        hashed=hashlib.md5(encoded)
        int_hashed=int(hashed.hexdigest(), 16)
        h = int_hashed % n
        finger_prints.append(h)
        if h in general_dict.keys():
            general_dict[h] + " , " + temp
        else:
            temp_dict={h:temp}
            general_dict.update(temp_dict)
#=====================================The end of the Pre-processing phase and also set of sentences=================================================



#=====================================Fingerprint set=====================================================================
in2=input('Press enter to see the answer to the first question of the fingerprints set\n')
# for the fingerprints part we have to chagne the epsilon between 10 ** -5 and 10 ** -6 to find the minimum number of bits  
#=========first question=========================

# in this section we try to find the right value for b and the right value for epsilon 
plagiarism_size= 6 #window size just like the Pre-processing phase 
verses = []
verse_count=0



for i in range (len(clean_words)):
    if len(clean_words[i:i+plagiarism_size])==plagiarism_size:
        verses.append(' '.join(clean_words[i:i+plagiarism_size]))

verses_set=set(verses)    
m = len(verses_set)
for j in range(10, 0, -1):
    epsilon = j * (10 ** -5)  # we have the  value for epsilon as mentioned in the question
    b = math.ceil(math.log((m/epsilon) , 2 )) # b has to be larger or equal to this value 
    n = 2 ** b # the value for n is equal to 2 to the power of b (as mentioned in the slides)
    finger_prints=[]
    for i in verses_set:
        temp=i
        encoded=temp.encode('utf-8')
        hashed=hashlib.md5(encoded)
        int_hashed=int(hashed.hexdigest(), 16)
        h = int_hashed % n
        finger_prints.append(h)

    finger_prints_set=set(finger_prints)
    if (len(finger_prints)-len(finger_prints_set) == 0):
        print (f'the value for epsilon is equal to  {epsilon} and the value of Bexp is equal to {b}\n')
        #here we get the right values for epsilon and also the right value for b 



#=================second question=====================
in3=input('Press enter to see the answer to the second question of the fingerprints set\n')

# Bteo ====> m = 1.17* sqrt(n) =>(m/1.17) ** 2 = 2 ** b => b = log((m/1.17 ** 2), base =2 ) = 33 (if we round it up)

temp =(m/1.17) ** 2
bteo = math.log(temp,2)
print(f'The theoratical value for b is equal to {round(bteo)}\n')
#we need to round it up since we need more bits in order to store stuff
# we can't use fewer bits in order to store them

#=================third question====================
in4=input('Press enter to see the answer to the third question of the fingerprints set\n')
#the formula used here is the formulas in the pages 14/15 of the slide
# epsilon = m / 2 ** b
m = len(verses_set)
bexp= 34 # found in question 1
new_epsilon = m/(2** bexp)
print(f' The theoratical value for epsilon with Bexp is equal to {new_epsilon}\n')

#the same value will be achieved if the formula is page 13 of the slide is used
#you can the code for that formula commented below
#########
# pr= 1- ((1-(1/(2**bexp))) ** m)
# print(pr)
#########
#===================4th question==============================
in4=input('The answer to the 4th question of the fingerprints set is available in the report in more details\n')
#The answer is Yes, Bteo is a good approximation of Bexp


#==============================size of the fingerprint set=======================
in5=input('Press enter to see the size of the fingerprints set using the value for Bexp\n')

#size of the fingerprint set with Bexp
#P(FP) is available in question 3
finger_prints_Bexp=[]
for i in verses_set:
    temp=i
    encoded=temp.encode('utf-8')
    hashed=hashlib.md5(encoded)
    int_hashed=int(hashed.hexdigest(), 16)
    h = int_hashed % n
    finger_prints_Bexp.append(h)
#asizeof function gives the size of the files in bytes         
print(f'The size of the fingerprint set with Bexp is equal to {asizeof.asizeof(finger_prints_Bexp)/1024} Kb\n')
#divided by 1024 to have values in KB
#====================The end of the fingerprints set part of the questions===========================




#======================Bit string array=====================================================================

#====================first question============================================
in6=input('Press enter to see the answer to the first question of the bit string array part\n')

#P(false positive) = (#ones/n)  based on what was said in the session of December 19th 2022
bit_string_array_sim_sizes_bytes={}
bit_string_array_sim_sizes_kb={}
m = len(verses_set) 
bits= [i for i in range(19,24)]
# n= [2**i for i in b]
# n = m = 2 ** b
General_dict={}
for b in bits:
    a=bitarray(2**b)
    a.setall(0)
    n=2**b
    FP_count = 0
    for i in verses_set:
        temp=i
        encoded=temp.encode('utf-8')
        hashed=hashlib.md5(encoded)
        int_hashed=int(hashed.hexdigest(), 16)
        h = int_hashed % n
        if a[h]==0:
            a[h]=1
        elif a[h]==1:
            FP_count+=1 # it actually checks the number of collisions 
    bit_string_array_sim_sizes_kb[n]={a.count(1)/n:asizeof.asizeof(a)/1024} #to have the sizes in kb and also the corresponding p(FP) value
    bit_string_array_sim_sizes_bytes[n]={a.count(1)/n:asizeof.asizeof(a)} #to have the sizes in bytes and also the corresponding p(FP) value
    General_dict[n] = a.count(1)/n # to calculate the value for the P(FP)



plt.figure(figsize=(10,10))
plt.plot([i for i in range(19,24)],General_dict.values())
plt.xlabel('Memory')
plt.ylabel('Epsilons')
plt.figtext(0.5, 0, "For memory, it's 2 to the power of what you see on the plot for the X axis", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.grid()
plt.show()
#by simulation
in111=input('Do you want to see the sizes of different bit string arrays? Y/N \n')
if in111=='y'or in111=='Y' or in111=="yes" or in111=='Yes':
    print(f'[n:[p(fb):sizes in kb]]{bit_string_array_sim_sizes_kb}')
    #{n:{P(FP): sizes in kb}} what you see in this dictionary
    #this dictionary contains the values for n , P(FP) and also the sizes in KB

#===============================Second question======================
in7=input('Press enter to see the answer to the second question of the bit string array part\n')

m = len(verses_set) 
b = [i for i in range(19,24)]
n= [2**i for i in b] 
# now based on the value of b and n we have to find different values for epsilon
epsilons = []
for i in n:
    epsilons.append(m/i)

plt.figure(figsize=(10,10))
plt.plot([i for i in range(19,24)],epsilons)
plt.xlabel('memory')
plt.ylabel('Epsilons')
plt.figtext(0.5, 0, "For memory, it's 2 to the power of what you see on the plot for the X axis", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.grid()
plt.show()
#based on theory
#======================================Third question=================
in7=input('Press enter to see the answer to the third question of the bit string array part\n')
plt.figure(figsize=(10,10))
plt.plot([i for i in range(19,24)],General_dict.values(), label = 'Simulation')
plt.plot([i for i in range(19,24)],epsilons,label='Theory')
plt.xlabel('Memory')
plt.ylabel('Epsilons')
plt.figtext(0.5, 0, "For memory, it's 2 to the power of what you see on the plot for the X axis", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.grid()
plt.legend()
plt.show()

# Both the simulation and the theory are close to each other

#======================================4th question============================================
in8=input('The answer to the 4th question of the bit string array part is available in the report in details\n')

#===========================================The end of the bit string array part of the questions===========================


#=========================================Bloom filters=====================================================================




#======================================first question=======================================================================
#first we find the optimal values for k (number of hashes)
# The values we get out of the formula are real numbers but we want integers
# We have to plot the performance graph in order to find out the right values for the number of hashes (K)
in9=input('Press enter to see the answer to the first question of the bloom filters part (It will take a while to make the performance graph)\n')
#finding the optimal values for k based on the theory
num_b =[j for j in range(19,24)]
m=len(verses_set)
n_num=[2**i for i in num_b]
hash_num_opt_theory=[(i/m)*math.log(2) for i in n_num]
plt.figure(figsize=(10,10))
plt.plot(n_num,hash_num_opt_theory)
plt.xlabel('Memory')
plt.ylabel('Number of Hashes')
plt.grid()
# plt.savefig('bloom_filters_question_one_hash_memory')
plt.show()




num_of_hash=[i for i in range(1,70)] #I chose a number more than 60 since there's also a k with value of 60.05019192818815
#but any value more than 61 would be unnecessary, so the value 70 is chosen only for the sake of a better visualization
bloom_filter_dict={}

for n in n_num:
    memory_dict={}
    for k in num_of_hash:
        a=bitarray(n) # a bitarray with lenght of  n
        a.setall(0) # setting all equal to zero
        false_count=0
        for i in verses_set:
            FP_count = 0
            for j in range(k):
                temp=i + str(j)
                encoded=temp.encode('utf-8')
                hashed=hashlib.md5(encoded)
                int_hashed=int(hashed.hexdigest(), 16)
                h = int_hashed % n
                if a[h]==0:
                    a[h]=1
                elif a[h]==1:
                    FP_count+=1 # it actually checks of ones so that we can compare them later with the value of k
            if FP_count==k:
                false_count+=1

        memory_dict[k] = (a.count(1)/n)**k
    bloom_filter_dict[n]=memory_dict
#P(FB) = (a.count(1)/m)**k 

#the performance graph to find the right values for k 

counter=19
plt.figure(figsize=(10,10))
for k,v in bloom_filter_dict.items():
    plt.plot(v.keys(),v.values(), label = f'2**{counter}')
    plt.xlabel('Number of Hashes')
    plt.ylabel('Epsilons')
    plt.grid()
    plt.legend()
    counter+=1
plt.show()
#now we can choose the right value based on the performance plot, the dictionary that contains the values for epsilon,
# and also the theoratical values for k that we have computed, the we can find the right values for k
# here we can see the obtained values for the optimal number of hashes from the formula
print (f'The theoratical values for the optimal number of hashes for each value for the memory (n) {hash_num_opt_theory}\n')
print('The right values for K has been chosen based on the performance graph and also the dictionary that contains the values for epsilon \n')
in333=input('Do you want to see the keys and values of the dictionary?Y/N\n')
if in333=='y'or in333=='Y' or in333=="yes" or in333=='Yes':
    print(bloom_filter_dict)
    #{n{different values for k : P(FP)}} what you see in this dictionary
''' 

This is how the right values have been chosen,
If you check the dictionary "bloom_filter_dict", you can see that for example for 2 ** 19, k = 4 has a lower value for false postive that 3, so for n=2 ** 19 the right value for k is 4
We do the same for the rest of the values
so the values were checked and the added to a dictionary (kopt) manually

'''
kopt={2**19:4,2**20:8,2**21:15,2**22:30,2**23:60}
print(f'The optimal values for k are : {kopt}\n')
#==========================second question==========================================
# the formula in page 29 of slide 12 is used 
# (1 - e ** -k*m/n) ** k
in10=input('Press enter to see the answer to the second question of the bloom filters part\n')

theoratical_epsilons={}
for n, k in kopt.items():
    theoratical_epsilons[n]={k:(1 - math.e ** ((-k*m)/n)) ** k}


print(f'you can see the theoratical values for the probability of false postive based on the optimal values for k here \n')
print('The values you see represent (in order) n , k(number of hashes), P(fp)\n ')
print(f'{theoratical_epsilons} \n')
#the list below is extracted from the theoratical_epsilons dictionary
epsilon_values_theory=[0.0744092833872199,0.0055367414541996,3.0253366974359995e-05,9.15266213285296e-10,8.377122411816051e-19]
'''
As we can see if we compare the value for the theoratical epsilons and the ones we got from the first question
The values are pretty close to each other
The first question is also a theory based question and we just used the value for the probability of False Positive in order to find the right values for k

'''

#===========================third question============================
in11=input('The answer to the 3rd question of the bloom filter part is available in the report in more details\n')
#=======================4th question==================================
in12=input('Press enter to see the answer to the 4th question of the bloom filters part\n')

# We have the optimal values for k
# We have different values for n and as a consequence different values for b
# we use the asizeof library and find the sizes and the plot them

bloom_filter_dict_sim={}
bloom_filter_dict_sim_kb={}
kopt
for n,k in kopt.items():
    memory_dict={}
    memory_dict_kb={}
    a=bitarray(n) # a bitarray with lenght of  n
    a.setall(0) # setting all equal to zero
    false_count=0
    for i in verses_set:
        
        FP_count = 0
        for j in range(k):
            temp=i + str(j)
            encoded=temp.encode('utf-8')
            hashed=hashlib.md5(encoded)
            int_hashed=int(hashed.hexdigest(), 16)
            h = int_hashed % n
            if a[h]==0:
                a[h]=1
            elif a[h]==1:
                FP_count+=1 # it actually checks of ones so that we can compare them later with the value of k
        if FP_count==k:
            false_count+=1

        memory_dict[k] = (a.count(1)/n)**k
        memory_dict_kb[k] = {(a.count(1)/n)**k:asizeof.asizeof(a)/1024} #to also have the sizes (sizes are equal to theory)

    bloom_filter_dict_sim[n]=memory_dict
    bloom_filter_dict_sim_kb[n]=memory_dict_kb #to also have the sizes (sizes are equal to theory)

#P(FB) = (a.count(1)/m)**k


#plotting 






# bloom_filter_dict_sim
#the values in the list below are gathered from the dictionary bloom_filter_dict_sim that is commented above 
epsilon_values_sim=[0.07441333677135287,0.005522559661331668,2.9930506277165827e-05,9.131377740901003e-10,8.263579823178387e-19]


plt.figure(figsize=(10,10))
plt.plot([i/(1024*8) for i in n_num],epsilon_values_sim, label = "simulation")
plt.xlabel('Memory (kb)')
plt.ylabel('False Postive')
plt.grid()
plt.legend()
plt.show()





in555=input('Do you want to see the dictionary containing the values for P(FP) and k and also n that were obtained from the simulation? Y/N \n')
if in555=='y'or in555=='Y' or in555=="yes" or in555=='Yes':
    print(f'[n : [k : [p(fp):size in kb]]]\n {bloom_filter_dict_sim}')


in222=input('Do you want to see the sizes of different bloom filters?Y/N\n')
if in222=='y'or in222=='Y' or in222=="yes" or in222=='Yes':
    print(f'[n : [k : [p(fp):size in kb]]]\n {bloom_filter_dict_sim_kb}')
    #{n:{k:{P(FP):sizes in kb}}} what you see in this dictionary
    # this dictionary contains the value for n , b , P(FP) and also the sizes in KB



#========5th question=========================

in13=input('Press enter to see the answer to the 5th question of the bloom filters part\n')


plt.figure(figsize=(10,10))
plt.plot([i/(1024*8) for i in n_num],epsilon_values_sim, label = "Simulation")
plt.plot([i/(1024*8) for i in n_num],epsilon_values_theory, label = "Theory")
plt.xlabel('Memory (kb)')
plt.ylabel('False Postive')
plt.grid()
plt.legend()
plt.show()

#As you can see in the plot, the values for epsilon gathered from simulation and theory are the same so the plots are the same 
# they overlap eachother
print('The plots overlap as you can see \n')
#=============The optional part====================
in14=input('Press enter to see the answer to the optional part\n')

# calculate the output of the formula for k from 1 to 10 and different values of n
# -1 * (n/k)* log(1-(a.count(1)/n),base=math.e)
#we're gonna calculte this number for hashes from 1 to 10 for different values of n

num_b =[j for j in range(19,24)]
m=len(verses_set)
n_num=[2**i for i in num_b]
num_of_hash=[i for i in range(1,10)] #I chose a number more than 60 since there's also a k with value of 60.05019192818815
#but any value more than 61 would be unnecessary
bloom_filter_dict_optional={}

for n in n_num:
    memory_dict_optional={}
    for k in num_of_hash:
        a=bitarray(n) # a bitarray with lenght of  n
        a.setall(0) # setting all equal to zero
        false_count=0
        for i in verses_set:
            FP_count = 0
            for j in range(k):
                temp=i + str(j)
                encoded=temp.encode('utf-8')
                hashed=hashlib.md5(encoded)
                int_hashed=int(hashed.hexdigest(), 16)
                h = int_hashed % n
                if a[h]==0:
                    a[h]=1
                elif a[h]==1:
                    FP_count+=1 # it actually checks of ones so that we can compare them later with the value of k
            if FP_count==k:
                false_count+=1

        memory_dict_optional[k] = -1 * (n/k)* math.log(1-(a.count(1)/n),math.e)
    bloom_filter_dict_optional[n]=memory_dict_optional
#P(FB) = (a.count(1)/m)**k lol
print(bloom_filter_dict_optional)



#=================The end of the bloom filters part===================================




