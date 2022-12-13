#importing the libraries 
import hashlib
import re
from pympler import asizeof
import math
import matplotlib.pyplot as plt


#############################################
#importing the txt file 
file = open("commedia.txt", "rt")
data = file.read()
############################################
#===========================
data=re.sub(r"[^\w\d'\s]+",'',data) # deleting the punctuations excpet for the apostrophes
words = data.split('\n') # spliting the data by each line 
del words[0:8]
clean_words=[]
for line in words.copy():
     #we use these conditions to remove the headers 
    if line.startswith("Inferno") or line.startswith("Purgatorio") or line.startswith("Paradiso")or not line.strip():
        words.remove(line) # deleting the headers 
    else: # if the line we are dealing with does not contain a header or anything 
        line = line.strip()
        splited_words = line.split(" ")
        clean_words.extend(splited_words)
print(f'the total number of words is equal to {len(clean_words)}  \n')
 #counting the number of words 
##############################################
print(f'Size of the words is equal to {asizeof.asizeof(clean_words)} Bytes \n')
#size of the words in bytes 
####################################
print(f'number of distinct words is equal to {len(set(clean_words))} \n')
######################################

# here we perform a sliding window like method to create the verses with the size of 4 or 8 
size_win= input('what is your desired length for verses? 8 or 4?')
if size_win=="4":
    plagiarism_size= 4 #window size
elif size_win=='8':
    plagiarism_size= 8 #window size 

verses = []
finger_prints=[]
verse_count=0



#here we create the verses 
for i in range (len(clean_words)):
    if len(clean_words[i:i+plagiarism_size])==plagiarism_size:
        verses.append(' '.join(clean_words[i:i+plagiarism_size]))

# these few lines are based on the formulas provided in the slides     
m = len(verses)
epsilon = 10 ** -4
b = math.log((m/epsilon) , 2 ) # b has to larger or equal to this value 
n = m * b

# here we perform the hashing process 
for i in range (len(verses)):
    temp=verses[i]
    encoded=temp.encode('utf-8')
    hashed=hashlib.md5(encoded)
    int_hashed=int(hashed.hexdigest(), 16)
    h = int_hashed % n
    finger_prints.append(h)


        
        
#we should store the fingerprints in a hashtable 
#then we should computer and store the finger prints inside of a hashtable 
print(f'\ntotal number of verses is {len(verses)}')
###########################################################
#we need to sotre everyyhing inside of a dictionary and the way I did it, both verses and the finger prints have to be created at the same time
#in order to compute "n" we need to create the verses, so the verses are created again in order for "n" to be computed 
verses_set=set(verses)
finger_prints_set=set(finger_prints)

print(f'len of verses before set {len(verses)} and after set {len(verses_set)} \n')
print(f'len of finger prints befor set {len(finger_prints)} and after set {len(finger_prints_set)}')
####################################################################################################
print(f'size of fingerprints is equal to {asizeof.asizeof(finger_prints_set)} bytes \n')
print(f'size of verses is equal to {asizeof.asizeof(verses_set)} bytes  \n')
####################################################################################################
#it's better to have both fingerprints and the verses stored in a dictionary 
# so we're gonna do it from the beginning
new_verses=[]
# m = len(verses) 
# we already have the value for m , b , epsilon and n since we've ran the code before
# epsilon = 10 ** -4
# b = math.log((m/epsilon) , 2 ) # b has to larger or equal to this value 
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

#####################################################################################################
ver_inp=input('do you want to see the verses? Y/N \n')
if ver_inp== 'Y' or ver_inp == 'y':
    window_stuff=input('what is your desired size for the verses? 4 or 8 \n')
    if window_stuff=='4':
        print(verses)
        print('\n')

    elif window_stuff=='8':

        for i in range (len(clean_words)):
            plagiarism_size=8
            if len(clean_words[i:i+plagiarism_size])==4:
                verses.append(' '.join(clean_words[i:i+plagiarism_size]))
        print(verses)
        print('\n')
########################################################################################################

############################################################################################################
# here's the plotting part, in this part we plot the theoratical outputs and we have to compare it to the size of the dictionary 

import numpy as np

epsilons=np.linspace(0,1,num=1000)
epsilons=list(epsilons)
epsilons.remove(0)
b_val=[]
m=len(verses)
for i in epsilons:
#     if i!=0:
    b_val.append(math.log((m/i) , 2))

        
plt.plot(epsilons,b_val)
plt.ylabel("b values ")
plt.xlabel('epsilons')
# plt.xscale('log')
plt.show()
# b_val[0:10]
###################################################################
mem=[]
for i in b_val:
    mem.append(m*i)
    
    
plt.plot(epsilons,mem)
plt.ylabel("memory")
plt.xlabel('epsilons')
# plt.xscale('log')
plt.show()
################################################################


print(f'The final size of the dictionary is equal to {asizeof.asizeof(general_dict)} bytes \n')

