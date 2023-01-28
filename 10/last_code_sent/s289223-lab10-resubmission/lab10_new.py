#importing the libraries 
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from scipy.stats import t
import os 
from statistics import mean 



seed=2004 # choosing a seed for random numbers 


#function to simulate the uniform distribution
def uniform_distribution_simulator():
    student_counter = 0  # the number of students in the year
    collision = 0 # occurrence of collisions
    days_of_the_year_uniform = [0] * 366
    while collision==0:
        birth_day = random.randint(0, 365)
        if days_of_the_year_uniform[birth_day] == 0:  # if there are no students born in that day of the year
            days_of_the_year_uniform[birth_day] = 1  # sets that day to one since a student is born in that day of the year
            student_counter += 1
        elif days_of_the_year_uniform[birth_day] == 1:  # we exit the while loop the moment we find a collision
            collision=1

    return student_counter  # returns the number of student it took until a collision happens




#adding the file used for realistic distribution

path_to_file = './US_births_1994-2003_CDC_NCHS.csv'
if os.path.isfile(path_to_file):
    column_names = ['year', 'month', 'date_of_month', 'day_of_week,births', 'births'] # adding the column names
    df = pd.read_csv(path_to_file, header=None, sep = ',', skiprows=1, index_col=None) # adding the file
    df.columns = column_names
# grouping the columns 'month', 'date_of_month' to provide the average number of people born on a specific day of the year
    average_births  = df.copy()
    month_day_group = average_births.groupby(['month', 'date_of_month'], as_index=False).mean()
else:
    print('please add the file for the realistic distribution of birthdays ')

#computing the statistics of the birthday data
sum_births = month_day_group['births'].sum()
month_day_group["prob"] = month_day_group['births'] / sum_births # finding the probabilty of each day
month_day_group["cdf"] = month_day_group["prob"].cumsum() # creading the cumulative distribution function (cdf)


# function to simulate the realistic distribution
def realistic_distribution_simulator():
    # Create a list of 366 zeros to represent the days of the year
    days_of_the_year_realistic = [0] * 366
    number_of_students = 0  # number of students it took until a collision happened
    while True:
        # we randomly find days of the year bassed on the value for cdf 
        randomly_generated_day = random.choices(month_day_group.index, weights=month_day_group['cdf'])[0]
    # If there are no students born on this day, increment the number of students and set the corresponding element in the list to 1
        if days_of_the_year_realistic[randomly_generated_day] == 0:
            days_of_the_year_realistic[randomly_generated_day] = 1
            number_of_students += 1
        # If there is already a student born on this day, set the collision flag to 1 to exit the while loop
        elif days_of_the_year_realistic[randomly_generated_day] == 1:
            break # collision has happened 
    # Return the number of students it took until a collision occurred
    return number_of_students


#performing the simulation 

#degree of freedom = n-1 in here it's 1999
np.random.seed(seed)
random.seed(seed) 
uniform_distro_conflicts=[]
realistic_distro_conflicts=[]
for i in range(0,1000):
    uniform_distro_conflicts.append(uniform_distribution_simulator())
    realistic_distro_conflicts.append(realistic_distribution_simulator())
#we get the average uniform conflicts on the left and the average real-life conflicts on the right
sum_of_conflicts={'Uniform distribution':sum(uniform_distro_conflicts)/len(uniform_distro_conflicts),'Realistic distribution':sum(realistic_distro_conflicts)/len(realistic_distro_conflicts)}



print( f"sum of conflicts for Uniform distribution: {sum_of_conflicts['Uniform distribution']} \n sum of conflicts for  Realistic distribution {sum_of_conflicts['Realistic distribution']}")

#evaluating the probabilites
# in this part we create classes of size 20 and compute the probabilites 

num_students_list = [i for i in range(100)]
uniform_prob = {}
realistic_prob = {}
num_simulations = 20 # 20 is the size of the class
for number_of_sutdents in num_students_list:
    
    class_uniform = [0] * num_simulations
    class_real_life = [0] * num_simulations
    for i in range(num_simulations):  
        uniform_bdays=[random.randint(0,365) for _ in range(0,number_of_sutdents)] # creating random birhtdays in uniform distro
        real_life_bdays=[int(realistic_distribution_simulator()) for _ in range(0,number_of_sutdents)] # creating random birthdays in realistic distro

        if len(uniform_bdays) != len(set(uniform_bdays)):
            class_uniform[i] = 1        
        # Simulate the real-life distribution
        if len(real_life_bdays) != len(set(real_life_bdays)): # checking for collision
            class_real_life[i] = 1

    uniform_prob[number_of_sutdents] = mean(class_uniform)
    realistic_prob[number_of_sutdents] = mean(class_real_life)


#plots


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
fig_df = pd.DataFrame(probabilities)
#graph probabilities
#the code above is used to create the theoratical plot 



plt.figure(figsize=(10,10))
plt.plot(uniform_prob.keys(), uniform_prob.values(), label='Uniform')
plt.plot(realistic_prob.keys(), realistic_prob.values(), label='Realistic')
plt.plot(fig_df['group_size'],fig_df['probability'],label='Theoretical')
plt.xlabel('Number of Students(m)')
plt.ylabel('Probability(birthday collision)')
plt.legend()
plt.grid()
plt.show()



#computing the confidence intervals 

#finding the confidence intervals 
mean_uniform=sum_of_conflicts['Uniform distribution']
mean_real_life=sum_of_conflicts['Realistic distribution']
uniform_std=np.std(uniform_distro_conflicts)
real_life_std=np.std(realistic_distro_conflicts)
degree_of_freedom=len(uniform_distro_conflicts)-1 #degree of freedom is equal to n-1 (len(uniform_conflict OR real_life_conflict)-1)
confidence=0.9 #(1-alpha)
critical_value = np.abs(t.ppf((1-confidence)/2,degree_of_freedom)) # have to use the t-student distro to find the confidence intervals
lower_bound_uniform=mean_uniform-uniform_std*critical_value/np.sqrt(len(uniform_distro_conflicts))
upper_bound_uniform=mean_uniform+uniform_std*critical_value/np.sqrt(len(uniform_distro_conflicts))
lower_bound_real=mean_real_life-real_life_std*critical_value/np.sqrt(len(realistic_distro_conflicts))
upper_bound_real=mean_real_life+real_life_std*critical_value/np.sqrt(len(realistic_distro_conflicts))
#confidence interval for uniform distribution
uniform_confidence_interval=(lower_bound_uniform,upper_bound_uniform )
#confidence interval for realistic distribution
realistic_confidence_interval=(lower_bound_real, upper_bound_real)  

print(f'the confidence interval for the realistic distribution is equal to: {realistic_confidence_interval} \n and for the uniform distribution is equal to: {uniform_confidence_interval} ')
#############################################################
user=input('Do you want to see the optional part? ')
if user=='y' or user == "Y":

        #running the code with higher m

    def uniform_distribution_simulator():
        student_counter = 0  # the number of students in the year
        collision = 0 # occurrence of collisions
        required_days = [0] * 1000
        while collision==0:
            birth_day = random.randint(0, 365)
            if required_days[birth_day] == 0:  # if there are no students born in that day of the year
                required_days[birth_day] = 1  # sets that day to one since a student is born in that day of the year
                student_counter += 1
            elif required_days[birth_day] == 1:  # we exit the while loop the moment we find a collision
                collision=1

        return student_counter

    uniform_conflicts = []
    num_students_list = [i for i in range(100)]
    uniform_prob = {}
    num_simulations = 25 # 20 is the size of the class
    for number_of_sutdents in num_students_list:
        
        class_uniform = [0] * num_simulations # which is the size of the class
        for i in range(num_simulations):  
            uniform_bdays=[random.randint(0,1000) for _ in range(0,number_of_sutdents)] # creating random birhtdays in uniform distro

            if len(uniform_bdays) != len(set(uniform_bdays)):
                class_uniform[i] = 1        

        uniform_prob[number_of_sutdents] = mean(class_uniform)
        #code to plot the theoretical distro
    numerator = 1000
    denominator = 1000
    #create empty list to store dictionary
    probabilities = []
    #for loop to generate probabilities
    for i in range(2, 100):
        numerator = numerator * (1000 + 1 - i)
        denominator = denominator * 1000
        probabilities.append({'group_size':i, 'probability':round(1 - (numerator / denominator), 3)})
        #print(round(probabilities, 3))
    #load dataframe
    df = pd.DataFrame(probabilities)


    print('performing the same experiment for m = 1000 on the uniform distribution and comparing it to the theoretical output')
    plt.figure(figsize=(10,10))
    plt.plot(uniform_prob.keys(), uniform_prob.values(), label='Uniform')
    plt.plot(fig_df['group_size'],fig_df['probability'],label='Theoretical')
    plt.xlabel('Number of Students(m)')
    plt.ylabel('Probability(birthday collision)')
    plt.legend()
    plt.grid()
    plt.show()    


