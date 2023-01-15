#==============================importing the libraries===============================

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar 
np.random.seed(1886)

#=================First part===================================
In= input("Press enter to see the first part of the simutaion which is for the Exponential-based h(t)")
#defining some of the required functions 
# based on the slides, s=s + thau (delta_t) and thau is from the same distro as h(t), h(t) can be uniform or exp
# we use h(t) and sigma to create the new time 
# h(t)s are the decay kernel function
      
def sigma(t): # sigma function as mentioned in the question
    return 20 * (t >= 0) * (t <= 10)
#uniform h(t)
def h_uniform(t):
    return np.random.uniform(0, 20) 
#exponential h(t)
def h_expo(t): 
    lambda_exp = 1/10
    return lambda_exp * math.exp(-lambda_exp * (lambda_exp * t))


# T = upper bound for time 
# decay = reproduction rate 
def hawkes_simulation(decay, T,h):
    s=0 # current time 
    dead_ppl=0#counts the number of dead people 
    dead_ppl_list=[] # to create the plot
    event_times=[]  # set of event times
    infected_ppl=1 #number of infected people, it should be equal to 1 in the beggining of the simulation
    last_infected=1 # the last number of infected people 
    infected_ppl_list=[]
    while s<T: # as long as our current time is less than the upper bound (100 days)
        if h == 1: #exponential
            intensity_at_s = sigma(s) + decay * sum( h_expo(s - t) for t in event_times) # we compute the intensity based on h(t) and sigma(t)
        elif h ==2: # uniform
                intensity_at_s = sigma(s) + decay*sum( h_uniform(s - t) for t in event_times) # computing the intensity 
        delta_t=np.random.uniform(0,5) # this is the time that will be added to the current time (it's notated as thau in the slides)
        s+=delta_t
        ra=np.random.uniform() # in order to add more randomness to the simulation
        if h ==1:
            intensity_prob = sigma(s) + decay * sum( h_expo(s - t) for t in event_times) # exponential
        elif h==2:
            intensity_prob = sigma(s) + decay*sum( h_uniform(s - t) for t in event_times) #uniform

        # if the condition below goes through, a new event will be created
        if ra < intensity_prob/intensity_at_s: # or D * intensity_at_s < intensity_prob
            infected_ppl+=1 # number of infected people will be increased
            dead_ppl = math.ceil(0.02 * infected_ppl) # we round it up
            infected_ppl_list.append(infected_ppl-dead_ppl) # we add the number of infected people - dead people to the list of infected people 
            # we subtract the number of dead people since they are dead and not infected anymore
            dead_ppl_list.append(dead_ppl)
            event_times.append(s) # we append the current time the list in which we store all the events
    if s<T: # at the end, if the 
        dead_ppl=math.ceil(0.02 * infected_ppl)  # to only get the number of dead people
        return event_times, dead_ppl, dead_ppl_list,infected_ppl_list
    elif s>=T:
        dead_ppl=math.ceil(0.02 * (infected_ppl-last_infected)) # to only get the number of dead people
        return event_times[:-1] , dead_ppl, dead_ppl_list[:-1],infected_ppl_list[:-1]







# function for plotting the first part 
def plot_events(event_times, infected_ppl_list, dead_ppl_list, ranges_list, ld):
    plt.figure()
    plt.plot(event_times, infected_ppl_list, label='Infected')
    plt.plot(event_times, dead_ppl_list, label='Death')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()
    # Plot the event times
    plt.scatter(event_times, np.zeros(len(event_times)))
    plt.xlabel('Time (in days)')
    plt.show()
    plt.figure(figsize=(10,2))
    plt.ylabel("Lambda * t")
    plt.yticks(np.arange(0, 5, 0.1))
    _ = plt.plot(ranges_list[12000:], ld[12000:], 'b-') # 12000 has been used for the sake of a better visualisation 
    plt.scatter(event_times,  np.zeros(len(event_times)))
    plt.xlabel('Time (in days)')
    plt.show()
        
        


# function to visualase the intensity (for the sake of a better understanding of the simulation)
def intensity_function_viz(event_times, T): 
    lambda_exp=1/10 # as mentioned in the question
    sample = np.asarray(event_times)
    ranges_list = np.arange(0, T, .001)  # creating  a list with the lenght of the upper bound of the time 
    ld=[]
    for i in ranges_list: #trying to calculate the intensity of the evenets so to make it possible to visualize 
        ld.append(sigma(i) + 2 * np.sum(lambda_exp * np.exp(-lambda_exp * (i - sample[sample < i]))))
    return ld, ranges_list 




# T= 100 # the upper bound for the  time of our simulation 
# decay=2 #reproduction rate        
# we start the simulation here 
event_times,dead_ppl,dead_ppl_list,infected_ppl_list=hawkes_simulation(decay = 2,T = 100,h=1)

#plots 
ld,ranges_list=intensity_function_viz(event_times, T=100)
plot_events(event_times, infected_ppl_list, dead_ppl_list, ranges_list, ld)


#=========================first part , uniform h(t)===================================================
in2=In= input("Press enter to see the first part of the simutaion for the Uniform-based h(t)")

#=========================== Everything is the same as above here but instead of an exponential h(t), a uniform h(t) is used =============================
# T= 100 # the upper bound for the  time of our simulation 
 # h = 2 to creat the uniform based h(t)
# decay=2 #reproduction rate

#starting the simulation
event_times,dead_ppl,dead_ppl_list,infected_ppl_list=hawkes_simulation(decay =2 ,T=100,h=2)
#============================================plotting the outputs=====================================================
ld,ranges_list=intensity_function_viz(event_times, T=100)
plot_events(event_times, infected_ppl_list, dead_ppl_list, ranges_list, ld)

#=========================================Second part===============================================
in3=input('Please press enter to see the second part of the simulation')

# these three commented functions are the same as the first part 
# def sigma(t):
#     return 20 * (t >= 0) * (t <=10)

# #uniform h(t)
# def h_uniform(t):
#     return np.random.uniform(0, 20) 
# #exponential h(t)
# def h_expo(t): 
#     lambda_exp = 1/10
#     return lambda_exp * math.exp(-lambda_exp * (lambda_exp * t))



# the function use to optimize the rho 
# rho will be used to minimize the cost 
def optimize_rho(cost, T, s, dead_ppl_list, max_death):
    # using the minimize_scalar from scipy to find the value of rho that minimizes the cost function
    rho = 1e5 * minimize_scalar(lambda rho: cost(rho) + (T-s)/T * (dead_ppl_list[-1]/s - max_death/T)**2, bounds=(0, 1), method='bounded').x
    return rho
# the cost fucntion as mentioned in the question 
def cost(c):
    return c ** 2

def hawkes_simulation_generalized(decay,T):
    dead_ppl_list=[] # to create the plot
    event_times=[]  # set of event times
    infected_ppl_list=[]
    cost_values=[]
    max_death=20000 # max number of dead people allowed 
    last_infected=1
    infected_ppl=1 #number of infected people, it should be equal to 1 in the beggining of the simulation
    s=0 # current time 
    dead_ppl=0#counts the number of dead people 
    rho=1 # the value for rho 
    initialization=False # a flag that checks whether we have passed time =20 or not 
    initialization_time=20
    while s<T:
#         intensity_at_s = sigma(s) + decay*sum( h_uniform(s - t) for t in event_times) # creating the current intensity
        intensity_at_s = sigma(s) + decay * sum( h_expo(s - t) for t in event_times)
        if s >= initialization_time:
            rho=optimize_rho(cost, T, s, dead_ppl_list, max_death)
            initialization=True     # it is equal to true since we have passed the time = 20   
        delta_t=math.ceil(np.random.uniform(0,30)) #the random number is generated between 0 and 30 in order to save time while running the code
        # you can change it to np.random.uniform(0, any number you want(smaller numbers will require more computation time))
        s+=delta_t # here the new time will be created , it's notated as thau in the slides
        ra=np.random.uniform()
#         intensity_prob = rho*(sigma(s) + decay*sum( h_uniform(s - t) for t in event_times))
        intensity_prob = rho*(sigma(s) + decay * sum( h_expo(s - t) for t in event_times))
        temp_prob=[] 
        if ra <= (intensity_prob/intensity_at_s):  # if this condition goes through, an even will happen (new infection)
            temp_prob=[np.random.poisson(decay) for i in range(last_infected)] # to create random numbers for each of child in the last generation
            # this number will be the number of infected people in the last generation 
            last_infected=sum(temp_prob) # we update the numer of last layer of the infeced tree of ppl
            infected_ppl+=last_infected
            dead_ppl=math.ceil(0.02 * infected_ppl) # we update the number of dead people 
            dead_ppl_list.append(dead_ppl)
            infected_ppl_list.append(infected_ppl-dead_ppl) # adding the number of infected people 
            event_times.append(s)
            if initialization==False:
                cost_values.append(0) # means there has been no cost 
            else:
                cost_values.append(cost_values[-1] + cost(rho)) # means that we have the cost
    if s<T: # if the last value of s is smaller than T, then we return the required variabels and lists 
        return event_times, dead_ppl, dead_ppl_list, infected_ppl_list, cost_values
            
    else: # if the value for s is larger than T, we remove the last value for the lists and then return the values for our variabels and lists 
        return event_times[:-1] , dead_ppl, dead_ppl_list[:-1],infected_ppl_list[:-1], cost_values[:-1]
   
# decay= 2
# T=365    days 

# we start the simulation here
event_times,dead_ppl,dead_ppl_list,infected_ppl_list,cost_values=hawkes_simulation_generalized(decay = 2,T= 365)





#============================================plotting the outputs for the second part=====================================================

plt.figure()
plt.plot(event_times, infected_ppl_list, label='Infected')
plt.plot(event_times, dead_ppl_list, label='Dead')
plt.xlabel('Time (in days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
#-----------------
plt.figure()
plt.plot(event_times, cost_values, label='Cost')
plt.xlabel('Time (in days)')
plt.legend()
plt.show()
#----------------------
# Plot the event times
# the zeros are use to create dots on the plot 
plt.scatter(event_times, np.zeros(len(event_times)))
plt.xlabel('Time (in days)')
plt.show()