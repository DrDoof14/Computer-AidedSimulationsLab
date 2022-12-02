#!/usr/bin/env python
# coding: utf-8

# <b>The inputs of the code are : <br></b>
# The length of the window (the length that we use to find the plagiarism)<br>
# The text file <br>
# <b>The outputs are: <br></b>
# the number of words and verses (just in this case)<br>
# the plagiarism output ( if there exists plagiarism or not)<br>
# 

# In[11]:


import re


# In[3]:


#reading the txt file 
# one of the ways used to import the the file
with open('commedia.txt','+r') as c:
    commedia=c.read()


# In[33]:


file = open("commedia.txt", "rt")
data = file.read()
#this part can also be used to import the file
#===========================
data=re.sub(r"[^\w\d'\s]+",'',data) # deleting the punctuations
words = data.split()
print('the number of words is \n')
print(len(words)) #counting the number of words 


# <b>words and verses</b><br>
# in order to define the number of verses we can use a moving window approach <br>
# the lenght of the window will be defined by the used and the window will move(slide) by one word <br>
# 

# In[20]:


plagiarism_size= 4 #window size 
verses = []
verse_count=0
for i in range (len(words)):
    verses.append(words[i:i+plagiarism_size])


# In[32]:


print('number of verses = 96704\n') 


# In[ ]:


print('the last index of the "verses list is 96704"')


# In[30]:


distintc_words = set(words)


# In[31]:


len(distintc_words)


# In[ ]:


print('total number of distinct words = 14684')


# The inputs of the code are :
# The length of the window (the length that we use to find the plagiarism)
# The text file
# The outputs are:
# the number of words and verses (just in this case)
# the plagiarism output ( if there exists plagiarism or not)
# 
# ###############################################
# 
# Code for reading the file
# 
# file = open("commedia.txt", "rt")
# 
# data = file.read()
# 
# ##################################################
# 
# total number of words = 96705
# 
# total number of verses = 96704 (it's not that efficient)
# 
# ##############################################
# 
# total number of distinct words = 14684
# 
# ##################################################### 
# 
# data structures 
# 
# a list of lists is used to store the verses 
# 
# also, a list is used to store the words
# 
# in a real-life software, a hashtable will be used 
# 
# #########################################
# 
# the general procedure 
# 
# in order to define the number of verses we can use a moving window approach
# the length of the window will be defined by the user and the window will move(slide) by one word

# In[ ]:




