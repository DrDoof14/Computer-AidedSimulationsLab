#dependencies
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from scipy.stats import t

user=input("Enter 1 for Task 1 (average)\n Enter 2 for Task 2 (probabilities)\n Enter 3 to see the comparison plot for probabilities (task 2)\n Enter 4 for confidence intervals \n Enter 5 for the optional part \n ")



#uniform distro simulation


def uniform():
    days=np.zeros(366) # we make a list of zeros with lenght of 366 which is the number of days 
    collision=0#number of collisions 
    flag = True
    counter=0#we need the number of people that enter the class until a conflict happens
    
    
    while flag == True:
        random_day=random.randint(0,365)
        if days[random_day]==0:
            days[random_day]=1
            counter=counter+1 #counts the number of students 
        else:
            flag = False
            
            
    return counter#number of students that came in the class before a conflict


#preparing the dataframe
column_name = ['year', 'month', 'date_of_month', 'day_of_week,births', 'births']
df = pd.read_csv('US_births_1994-2003_CDC_NCHS.csv', header=None, sep = ',', skiprows=1, index_col=None)
df.columns = column_name


# group the 'month', 'date_of_month' columns to produce average number of people born on specific day
temp = df.copy()
month_day_group = temp.groupby(['month', 'date_of_month'], as_index=False).mean()


# computing the distribution of the birthday data

sum_births = month_day_group['births'].sum()
month_day_group["prob"] = month_day_group['births'] / sum_births
month_day_group["cdf"] = month_day_group["prob"].cumsum()

# defining the function in order to create a random day [1, 366] from the real distribution
#we are creating the cdf 
def real_life_birth_cdf():
    u = np.random.uniform(0, 1)
    for day, cdf in enumerate(month_day_group['cdf']):
        if u <= cdf:
            return (day)
        
#works pretty simple, creates a random int and if cdf is bigger than the random int which is our threshold we 
#will return the day


def real_life():
    days=np.zeros(366) # we make a list of zeros with lenght of 366 which is the number of days 
    collision=0#number of conflict 
    Flag = True
    counter=0#we need the number of people that enter the class until a conflict happens
    #it's the as the uniform distro 
    while Flag == True:
        
        random_day=real_life_birth_cdf()
        if days[random_day] ==0:
            days[random_day] = 1
            counter=counter+1
        
        else:
            Flag= False


    return counter



if user == "1" or user =="4":
    #defree of freedom = n-1 in here it's 1999
    seed=1886
    np.random.seed(seed)
    random.seed(seed) 
    uniform_conflict=[]
    real_life_conflict=[]
    for i in range(0,1000):
        uniform_conflict.append(uniform())
        real_life_conflict.append(real_life())
    #we have the average uniform conflicts on the left and the average real-life conflicts on the right
    sum_of_conflicts={'Uniform Distro':sum(uniform_conflict)/len(uniform_conflict),'Real-life':sum(real_life_conflict)/len(real_life_conflict)}
    if user == "1":
        print('In theory: for n = 365, m ≈ 22.3 and E[m] = 23.9 and the emprical E[m] is as shown below\n{}'.format(sum_of_conflicts))
    elif user == "4":
        mean_uniform=sum_of_conflicts['Uniform Distro']
        mean_real_life=sum_of_conflicts['Real-life']
        uniform_std=np.std(uniform_conflict)
        real_life_std=np.std(real_life_conflict)
        dof=999 #degree of freedom is equal to n-1 (len(uniform_conflict OR real_life_conflict)-1)
        confidence=0.9 #(1-alpha)
        t_crit = np.abs(t.ppf((1-confidence)/2,dof))
        #confidence interval for uniform distro
        uniform_confidence_interval=(mean_uniform-uniform_std*t_crit/np.sqrt(1000), mean_uniform+uniform_std*t_crit/np.sqrt(1000)) 
        #confidence interval for real distro
        real_life_confidence_interval=(mean_real_life-real_life_std*t_crit/np.sqrt(1000), mean_real_life+real_life_std*t_crit/np.sqrt(1000)) 
        print('the confidence interval for the real-life distro is equal to: {} \nand for the uniform distro is equal to: {} '.format(real_life_confidence_interval,uniform_confidence_interval))

elif user == "2" or user =="3":
    list_of_students=[i for i in range(0,100)] #based on our distro, we better not use anything more than 100
    uniform_prob={}
    real_life_prob={}
    for f in list_of_students:
        class_uniform=[0]*20
        class_real_life=[0]*20    #we create classes both for the real-life scenario and also the uniform scenario

        for d in range(len(class_real_life)):
            uniform_bday=[random.randint(0,365) for _ in range(0,f)]
            real_life_bday=[int(real_life()) for _ in range(0,f)]
            # checking if there is a conflict in one specific class and make it 1
            if len(uniform_bday) != len(set(uniform_bday)):
                class_uniform[d] = 1
                    
            if len(real_life_bday) != len(set(real_life_bday)):#set removes all the duplicate, so if there's something similiar

                class_real_life[d] = 1
    #probablities
        uniform_prob.update({f:(sum(class_uniform)/len(class_uniform))})
        real_life_prob.update({f:(sum(class_real_life)/len(class_real_life))})
#we have to consider that the number of students should be higher than 2 (so the conflict happens)
#and the number of trials should be higher than 32 (based on the centeral limit theorem)
    if user == "2":
        print('probabilities for uniform distribution: {}\n probabilities for real-data distribution: {}'.format(uniform_prob,real_life_prob))
    elif user == "3":
        #code to plot the theoretical distro
        numerator = 366
        denominator = 366
        #create empty list to store dictionary
        probabilities = []
        #for loop to generate probabilities
        for i in range(2, 100):
            numerator = numerator * (366 + 1 - i)
            denominator = denominator * 366
            probabilities.append({'group_size':i, 'probability':round(1 - (numerator / denominator), 3)})
            #print(round(probabilities, 3))
        #load dataframe
        df = pd.DataFrame(probabilities)
        plt.plot(uniform_prob.keys(), uniform_prob.values(), label='uniform')
        plt.plot(real_life_prob.keys(), real_life_prob.values(), label='realistic')
        plt.plot(df['group_size'],df['probability'],label='theoretical')
        plt.xlabel('Number of Students(m)')
        plt.ylabel('Prob(birthday collision)')
        plt.legend()
        plt.show()
if user == '5':
    pass 

