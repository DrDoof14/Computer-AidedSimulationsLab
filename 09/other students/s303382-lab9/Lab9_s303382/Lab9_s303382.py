import matplotlib.pyplot as plt
import statistics as s
from numpy import random
import numpy as np
import scipy.stats as st
from scipy.stats import t
import math



#Definitions of variables and a function for the hyper-exponential case.
#These parameters have been chosen empirically, solving the system of equations given by the 2 constraints: mean = 1 and standard deviation equal to 10. 
p1 = 0.005
p2 = 0.995
lambda1 = (3*math.sqrt(4378)-2)/19699
lambda2 = (1/299)*(398+3*math.sqrt(4378))

def generate_hyper(prob_list = [p1,p2], expectations = [1/lambda1,1/lambda2]):
    u = np.random.uniform(0,1)
    if u<=prob_list[0]:
        x = np.random.exponential(expectations[0])
    else:
        x = np.random.exponential(expectations[1])
    return x

#Defining the classes Client and Queue

class Client:
    def __init__(self,arrival):
        self.arrival_time = arrival
class Queue:
    def __init__(self , utilization,service_time_type):
        self.users_waiting = 0       #Number of users waiting to be served 
        self.users = []              #Clients in the system (list of objects of type Client)    
        self.FES = [(0,"arrival")]   #The inizialization of the FES is done scheduling the first arrival at time t = 0
        self.delays = []             #List of the delays for each custome
        self.service_type = service_time_type
        self.ARRIVAL_MU = 1 / utilization   #The utilization is lambda(A) * E[S]. In the 3 cases the mean of the service time is always = 1
    
    def sample_service_time(self):
        if self.service_type == 'EXP':
            return random.exponential(1)
        elif self.service_type == 'DET':
            return 1
        else:
            return generate_hyper()
            
    
    def schedule_arrival(self,time):
        #Sample the time until next arrival 
        inter_arrival = random.exponential(self.ARRIVAL_MU)
        #Scheduling the next arrival (at time current time + interarrival)
        self.FES.append((time+inter_arrival,"arrival"))
    
    def schedule_departure(self,time):
        #Sample the service time
        service_time = self.sample_service_time()
        #Schedule when the client will depart
        self.FES.append((time+service_time,"departure"))
     
    def arrival(self,time):
        #Creating the client who arrived
        c = Client(time)
        #Adding the client in the queue
        self.users.append(c)
        #Updating the state of the system
        self.users_waiting+=1    
        #Scheduling next arrival
        self.schedule_arrival(time)
        #If the server is idle, start the service
        if len((self.FES)) == 1 and self.users_waiting > 0:
            #A client start to be served, so I update the number of clients waiting in line. 
            self.users_waiting -= 1
            #Scheduling the departure of the client
            self.schedule_departure(time)
    
    def departure(self,time):
        #Removing the client who departed
        user_departed = self.users.pop(0)
        #Checking whether there are more clients to serve in the line, if yes start the service
        if self.users_waiting>0:
            #A client started to be served, so I update the number of clients waiting in line. 
            self.users_waiting -= 1
            #Scheduling the departure of the client
            self.schedule_departure(time)
        #Returning the client who departed in order to compute the delay (time between arrival and departure)
        return user_departed        
    
    def update_delays(self,delay):
        self.delays.append(delay)

#Definitions of the utils function
  
def queue_event(queue):               #This function models the occouring of an event in the queue
    #Sorting the FES according to time
    queue.FES.sort(key= lambda tup: tup[0], reverse = True)
    #Extracting the event with the smallest time
    (time, event_type) = queue.FES.pop()
    #Checking if the event extracted is an arrival or a departure and calling the associated function of the queue
    if event_type == "arrival":
        queue.arrival(time)
    elif event_type == "departure":
        user_departed  = queue.departure(time)      #Extracting the client who departed   
        delay = time - user_departed.arrival_time   #Computing the delay for such a client as departing time minus arrival time 
        queue.update_delays(delay)                  #Updating the list of the delays of the queue

def select_dim_batch(utilization):    #This function select the dimension of the batches according to the value of utilization
    if utilization < 0.4:
        return 500
    elif utilization <= 0.8:
        return 700
    elif utilization <= 0.95:
        return 1000
    else:
        return 2000

def compute_cumulative_mean(list):    #This function returns the cumulative mean of the list passed as input parameter
    cum_mean = []
    acc = 0.0
    for idx,x_i in enumerate(list):
        acc += x_i
        cum_mean.append(acc/(idx+1))
    return cum_mean 

def compute_CI_batch_means(data,confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)      #1-alfa/2
    return m,h

def check_transient(means):      
    #print("Condizione transiente "+str(max(means[1]/means[0], means[0]/means[1])))
    if max(means[1]/means[0], means[0]/means[1]) > 1.15:
        return False
    else:
        return True

def plot_transient(delays,transient,utilization,service):
    x = [i for i in range(len(delays))]
    y = compute_cumulative_mean(delays)
    plt.figure()
    plt.title("Utilization "+str(utilization)+" || Service Time: "+str(service))
    plt.plot(x,y,color ='b')
    plt.axvline(x = transient, color = 'r')
    plt.xlabel('N (Client Index)')
    plt.ylabel("Cumulative mean of the delays")
    plt.show()
    
def plot_average_delays(av_delays,ci,utilizations,service):
    x = utilizations
    y = av_delays
    plt.figure()
    plt.title("Service Scenario = "+str(service))
    plt.plot(x,y,color ='r')
    plt.fill_between(x, [yi-c_i for yi,c_i in zip(y,ci)], [yi+c_i for yi,c_i in zip(y,ci)], color='b', alpha=.1) 
    plt.grid()
    plt.xlabel('Utilization')
    plt.ylabel("Average Delay")
    plt.show()

#def plot_comparison(R1,R2,R3,utilizations):

#Definition of the simulator function. Note that all the utils function are used in the simulator and the simulator is called only once inside the main. 
def simulator(NUMITER,utilizations,service):
    minimum_number_batches = 10                         #Minimum number of batches to be considered in the context of the batch means technique
    average_delay_values = []                           #This list will contain the values of average delay, one for each utilization
    ci = []                                             #This list will contain the semi-widths of the CI's related to the average delays

    #Starting the simulation
    for utilization in utilizations:
        DIM_BATCH = select_dim_batch(utilization)       #Selecting the dimension of the batches both for transient identification and batch means technique
        n_iter = 0                                      #Number of current iterations
        queue = Queue(utilization,service)              #Creating the queue system
        
        #Variables for the transient identification
        transient_identified = False                    #True if the transient has been identified
        transient_value = 0                             #The value of n associated to the end of the transient and the beginning of the steady-state
        batch_transient_counter = 0                     #Count the number of batches analyzed for the transient identification
        mean_batches_transient_id = []                  #This list will contain the mean of the current and then previous batch. It will contain max 2 elements.
        
        #Variables for the batch means 
        batch_means = []                                #This list will contain the means of the batches in the context of batch means method      
        batch_means_counter = 0                         #Count the number of batches analyzed for the batch means
        #ok_batch_means = False                         #True if a feasibile number of batches has been identified for the batch means technique
        confidence_interval = (0,0)
        accuracy = 0.0
        #The loop begins
        while n_iter < NUMITER:
            
            if not transient_identified: #Transient Identification
                
                if len(queue.delays) == DIM_BATCH*(batch_transient_counter + 1) and len(queue.delays)!=0:    #If I have enough data to identify a batch
                    batch_transient_counter += 1
                    mean_batch = s.mean(queue.delays[len(queue.delays) - DIM_BATCH :])                       #Computing the mean of the current batch
                    mean_batches_transient_id.append(mean_batch)                                             #Storing such mean in the list mean_batches_transient_id
                    
                    if len(mean_batches_transient_id)>1:                                                     #If I have the mean of more than one batch
                        transient_identified = check_transient(mean_batches_transient_id)                    #Checking if I can identify the transient 
                        
                        if transient_identified == True:                                                     #If I have identified the transient
                            transient_value = len(queue.delays) - 1                                          #Storing the value of n for which I have the end of the transient and the beginning of the steady state. 
                             #Debugging: len(delay) must be equal to transient_value here
                        else:
                            mean_batches_transient_id.pop(0)                                                 #If the transient has not been identified I remove the mean of the previous batch, i.e. I loose infos about the previous batch, is not useful anymore.  
            else: #Batch means
                
                if len(queue.delays) == transient_value + DIM_BATCH*(batch_means_counter + 1):      #Starting from the transient value, if i have enough data to identify a batch ...
                    batch_means_counter += 1
                    mean_current_batch = s.mean(queue.delays[len(queue.delays) - DIM_BATCH  :])
                    batch_means.append(mean_current_batch)                                                                 #Computing the mean inside the current batch
            
                    if(batch_means_counter >= minimum_number_batches):                                                     #If I have enough batches (at least 10)
                        avg_batches, width_CI = compute_CI_batch_means(batch_means,0.95)                                   #Computing the confidence interval for the mean. Here i compute the mean of the means of the batches and I evaluate the confidence interval   
                        if width_CI/avg_batches <= 0.2:                                                                    #Checking the accuracy of the technique. If not, increase the number of batches and go on...
                            confidence_interval =  (avg_batches-width_CI,avg_batches+width_CI)                             #Computing the confidence interval for the considered batches
                            average_delay_values.append(avg_batches)                                                       #Appending the average of the batches inside the list that will contain one value for each utilization. For the last point of the exercise "computing 95% confidence intervals in function of the utilization"
                            ci.append(width_CI)
                            accuracy = width_CI/avg_batches                                                                #Appending the semiwidth of the interval for the same reason
                            #ok_batch_means = True                                                                          #Changing the flag to True
                            break
                    if(batch_means_counter == 30):                                                                         #If I have 30 batches and still do not converge just compute the confidence interval as well
                        confidence_interval =  (avg_batches-width_CI,avg_batches+width_CI)
                        ci.append(width_CI)
                        average_delay_values.append(avg_batches)
                        accuracy = 'Batch means didnt converge'
                        break         
            
            queue_event(queue)
            n_iter+=1
        
        #Printing and plotting the results
        print("Utilization = "+str(utilization)+" || Dim_Batch = "+str(DIM_BATCH)+\
              " || "+"Transient value = "+str(transient_value)+" || "+ "len(delay) at the end of batch means = "+str(len(queue.delays)))
        plot_transient(queue.delays,transient_value,utilization,service)
        print("Utilization = "+str(utilization)+" || Dim_Batch = "+str(DIM_BATCH)+\
              " || "+"Number of batches batch means = "+str(batch_means_counter)+" || "+ "Accuracy = "+str(accuracy)+\
              " || "+ "Confidence Interval = "+str(confidence_interval))
    
    plot_average_delays(average_delay_values,ci,utilizations,service)
    return (average_delay_values,service)
    
    
    
    
# ****************** MAIN ******************

if __name__ == '__main__':
    np.random.seed(10)
    #Calling the simulator. Note that here you can change the values of the input parameters of the simulation 
    simulator(NUMITER = 500000, utilizations = [0.1, 0.2, 0.4,0.5, 0.7, 0.8, 0.9, 0.95, 0.99], service = 'DET')
    simulator(NUMITER = 500000, utilizations = [0.1, 0.2, 0.4,0.5, 0.7, 0.8, 0.9, 0.95, 0.99], service = 'EXP')
    simulator(NUMITER = 500000, utilizations = [0.1, 0.2, 0.4,0.5, 0.7, 0.8, 0.9, 0.95, 0.99], service = 'HYP')