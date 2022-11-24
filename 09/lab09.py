import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import numpy as np
import pandas as pd
import math
import dataframe_image as dfi 



pars1= input("to see all the distributions in a plot enter 1 and if you want to see them separately enter 2: \n It will take a while for the code to run, so please be patient. \n")
pars2= input("Enter Y/N if you want to see the numerical info of the simulation in a PNG file: \n")

def arrival(time, FES, queue, average_arrival_time, average_service_time):
    
    global users
    global customer
    
    # introducing random client arrival
    inter_arrival = random.expovariate(1.0/average_arrival_time) #this part will always be expovariant and we don't change it  
    
    #FES.put((time + inter_arrival, 'arrival'))
    FES.append((time + inter_arrival, 'arrival'))
    
    # managing the event of customers coming to the post office 
    users += 1
    x = 'client' + str(customer)
    customer += 1
    
    # recording client id and put it in the list for further use 
    client = Client(x, time)
    queue.append(client)

    print(f'{client.name} arrived at {client.arrival_time}')
    
    # start the service in case the server is idle
    if users == 1: #if it's the first user
        # scheduling random departure time to the clients
        service_time = average_service_time
        FES.append((time + service_time, 'departure'))


def departure(time, FES, queue, average_arrival_time, average_service_time):
    
    global users
    
    # manipulating the list of clients to get FIFO orientation
    queue.reverse() # we first reverse the queue since it's  First In First Out 
    client = queue.pop() # we give service to the customer who arrived earlier 
    queue.reverse() # then we again reverse the queue to go back to the previous state
    users -= 1 # we reduce the number of users that are waiting 
    delay = time - client.arrival_time # we computer the delay for the next user 
    
    print(f'{client.name} departured at {time}')
    
    # checking the number of clients in line
    if users > 0:
        # scheduling random departure time to the clients
        service_time = average_service_time  # we do not change the distro of the time in these functions 
        FES.append((time + service_time, 'departure'))
    
    return delay

class Client:
    def __init__(self, name, arrival_time):
        self.name = name
        self.arrival_time = arrival_time
        


 # Finding the transient point (k)



def transient_point(cumulative_delay, u): #u is the utilization
    
    ave = np.mean(cumulative_delay)
    std = np.std(cumulative_delay)
    
    if u <= 0.6: #  we choose a value such as 0.6 or maybe 0.7 in order to avoid the transient being negative
        j = int(len(cumulative_delay) * u)
    else:
        j = int(len(cumulative_delay) * 0.6) #when we have higher u, we have problems with indexed
        # so a number such as 0.6 was chosen to avoid the problem
    
    for i in range(j, len(cumulative_delay)):
        if (cumulative_delay[i] > ave - std) and (cumulative_delay[i] < ave + std):
            return i



def confidence_interval_margin(batch_mean_list, confidence_interval): # gives the margin for the confidence interval
    """
    first we compute the mean for each batch and then after that we have a list of means 
    then we compute the mean of the numbers in that list in order to compute the confidence interval 
    we also find the std of that list and use it in order to compute the confidence interval 
    
    """
    mu = np.mean(batch_mean_list)
    std = np.std(batch_mean_list)
    n = len(batch_mean_list)
    t = np.abs(ss.t.ppf((1-confidence_interval)/2, n - 1)) # we use the t from t student
    
    margin = t * std / np.sqrt(n) # as shown in page 16 of slide 10
    
    expanding_condition = 2 * margin / mu # as shown in page 16 of slide 10
    """
    the function returns the mean of the means (mu) which is computed based on the mean of the each batch
    the margin
    and the expanding condition which will be used in further steps
    """
    
    return mu, margin, expanding_condition 


def hyper_expo():

    
    # return service
    p = .5 # we chose a probabilty equal to accuracy we chose, which is 0.5
    w1 = 1/6 # we define two different winning probabilities
    w2 = 1/8 
    u = random.random() #we can aslo use np.random
    """
    it's like tossing a coin, we define a threshold and if the random number is lower than that player 1 wins 
    if not,player 2 wins
    """
    if u <= p: 
        expec = w1
    else:
        expec = w2
    service = random.expovariate(1/expec)
    
    
    # another way of writing this function 
    # p = .5
    # l1 = 1/1.5
    # l2 = 1/1.1
    # u = random.random()
    # v = random.random()
    # if u <= p:
    #     service = -1.5 * math.log(1 - v)
    # else:
    #     service = -1.1 * math.log(1 - v)
    
    return service


#The value for P should be set by user and we put 0.5



np.random.seed(42)
random.seed(42)

simulation_time = 50000
simulation_time_warm_up = 10000 #this time is defined to find the transient
#we run the code to the point of finding the transient and then we halt the process and find the K  

utilization = [0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99]
distributions = ['deterministic', 'exponential', 'hyperexponential'] # three different distributions

dict_simulation = {}

accuracy = 0.5 # which is denoted as P in the formula in page 16 of the slide 10

for u in utilization:
    
    for distribution in distributions:
        
        time = 0 # the initial point of our time 
        users = 0
        customer = 1
        queue = []
        FES = []
        delay = []
        
        average_arrival_time = 1/u 
        confidence_interval = 0.95 # as mentioned in the question
        
        cumulative_delay = []
        variance_delay = []
        #we also use variance in order to find the transient point
        
        batch_mean_list = []
        expanding_number = 0 # it is the size of the batch that will be added each time after expanding (when the condition is not met and we have to keep on working)
        batche_initial_size = 10
        batch_count = 10 #it surely shouldn't be lower than 10
        flag =1 #it is used to end the simulation
        
        FES.append((0,'arrival'))
        target = 0
        
        
        while time < simulation_time and flag:
#             # average_service_time = random.expovariate(1) 

#             average_service_time = hyper_expo()
            
#             # Lambda = u/average_service_time
#             # average_arrival_time = 1/Lambda
            
            if distribution == 'deterministic':
                average_service_time = 1
            elif distribution == 'exponential':
                average_service_time = random.expovariate(1)
            else:
                average_service_time = hyper_expo()
            
            
            FES = sorted(FES)
            (time, event_type) = FES[target]
            
            if event_type == 'arrival':
                arrival(time, FES, queue, average_arrival_time, average_service_time)
            elif event_type == 'departure':
                delay.append(departure(time, FES, queue, average_arrival_time, average_service_time))
                cumulative_delay.append(sum(delay)/len(delay))
                # the delay happens in departure and taht's why it's been added here
            
            if time > int((simulation_time_warm_up * (1 + u) + expanding_number)): # if there's still time to go/ data to deal with/ haven't reached the end of the data we have
                # we add the simulation time by 1+u so that u affects the added simulation time and as a consequence 
                # it affects the number and also the size of batches
                
                if expanding_number == 0: #we haven't added batches yet and we haven't decided the size of the newly added batches yet 
                    k = transient_point(cumulative_delay, u)
                    batch_size = int(((simulation_time_warm_up * (1 + u) - k)/batche_initial_size))
                    batch_start_index = k #the first index after finding the transient is the first index of the batches 
                else:
                    batch_start_index = len(cumulative_delay) - batch_size
                
                
                while(batch_start_index < len(cumulative_delay)): 
                    batch_mean_list.append(np.mean(cumulative_delay[batch_start_index:(batch_start_index + batch_size)])) # we compute the mean of the batches
                    batch_start_index += batch_size # we add the end of the last batch as the initial point of the next batch
                
            
                mu, margin, expanding_condition  = confidence_interval_margin(batch_mean_list, confidence_interval)
                
                if expanding_condition > accuracy: # the scenario in which we add batches to the simulation and keep on simulating 
                    expanding_number += batch_size
                    batch_count += 1 # number of batches added
                else:
                    flag = 0 # used to exit the simulation
                    
            target += 1
        # at the end we store all the outputs of the simulation on different distributions and different utilization values
        dict_simulation[(u, distribution)] = {'k': k, 'last_mu': mu, 'ci': (mu - 2 * margin, mu + 2 * margin), 'batch_count': batch_count, 'batch_size': batch_size, 'cumulative_delay': cumulative_delay}
          
        


 # Gaining info on the data frame

df = pd.DataFrame.from_dict(dict_simulation)
# df


# to store the information describing the data frame 
if pars2 == "Y" or pars2 == "y":
    df_head=df.head(6)
    dfi.export(df_head, 'df_head.png',max_cols=-1)


# # Plotting



# in this part of the code plots the average delays based on the different values for utilization for three different distributions
# the confidence intervals are also shown in on the plots

if pars1 == "1":
    last_mu_d,last_mu_e, last_mu_h=[],[],[]
    ci_list_d,ci_list_e,ci_list_h=[],[],[]
    for u in utilization:
        last_mu_h.append(df[(u, 'hyperexponential')]['last_mu'])
        last_mu_e.append(df[(u, 'exponential')]['last_mu'])
        last_mu_d.append(df[(u, 'deterministic')]['last_mu'])
        ci_list_d.append(df[(u, 'deterministic')]['ci'])
        ci_list_e.append(df[(u, 'exponential')]['ci'])
        ci_list_h.append(df[(u, 'hyperexponential')]['ci'])

        
    d0,d1,e0,e1,h0,h1=[],[],[],[],[],[]

    for i in ci_list_d:
        d0.append(i[0])
        d1.append(i[1])
    for i in ci_list_e:
        e0.append(i[0])
        e1.append(i[1])
    for i in ci_list_h:
        h0.append(i[0])
        h1.append(i[1])
        

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(utilization, last_mu_e, label="Exponential")
    ax.plot(utilization, last_mu_d, label="Deterministic")
    ax.plot(utilization, last_mu_h, label=" Hyper-exponential")
    ax.fill_between(utilization, e0, e1, alpha=0.5)
    ax.fill_between(utilization, d0, d1, alpha=.5)
    ax.fill_between(utilization,h0,h1, alpha=.5)
    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel="Utilisation")
    ax.set_ylabel(ylabel="Delay")
    ax.set_title("Delays and Utilisations")
    plt.savefig('final')
    plt.show()


#  separately showing different plots for different distributions with confidence intervals

# Deterministic

if pars1 == "2":

    last_mu_d=[]
    ci_list_d=[]
    for u in utilization:
        last_mu_d.append(df[(u, 'deterministic')]['last_mu'])
        ci_list_d.append(df[(u, 'deterministic')]['ci'])
        
    d0,d1=[],[]

    for i in ci_list_d:
        d0.append(i[0])
        d1.append(i[1])

        

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(utilization, last_mu_d, label="Deterministic")
    ax.fill_between(utilization, d0, d1, alpha=.5)
    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel="Utilisation")
    ax.set_ylabel(ylabel="Delay")
    ax.set_title("Delays and Utilisations for the Deterministic distribution")
    plt.savefig('Deterministic')
    plt.show()
    

# Exponential

if pars1 == "2":
    last_mu_e=[]
    ci_list_e=[]
    for u in utilization:
        last_mu_e.append(df[(u, 'exponential')]['last_mu'])
        ci_list_e.append(df[(u, 'exponential')]['ci'])

        
    e0,e1=[],[]


    for i in ci_list_e:
        e0.append(i[0])
        e1.append(i[1])


    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(utilization, last_mu_e, label="Exponential")

    ax.fill_between(utilization, e0, e1, alpha=0.5)

    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel="Utilisation")
    ax.set_ylabel(ylabel="Delay")
    ax.set_title("Delays and Utilisations for the Exponential distribution")
    plt.savefig('Exponential')
    plt.show()


# hyper-exponential

if pars1 == "2":


    last_mu_h=[]
    ci_list_h=[]
    for u in utilization:
        last_mu_h.append(df[(u, 'hyperexponential')]['last_mu'])
        ci_list_h.append(df[(u, 'hyperexponential')]['ci'])

        
    h0,h1=[],[]


    for i in ci_list_h:
        h0.append(i[0])
        h1.append(i[1])
        

    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(utilization, last_mu_h, label=" Hyper-exponential")

    ax.fill_between(utilization,h0,h1, alpha=.5)
    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel="Utilisation")
    ax.set_ylabel(ylabel="Delay")
    ax.set_title("Delays and Utilisations for the Hyper-exponential distribution")
    plt.savefig('hyper-exponential')
    plt.show()






