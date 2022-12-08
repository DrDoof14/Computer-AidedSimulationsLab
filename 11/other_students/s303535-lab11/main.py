import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sympy import symbols, Eq, solve
import os

np.random.seed(4899)

class Client:   #class of the object client
    def __init__(self, arr_time, priority,service_time_left,server_serving=None):
        self.arr_time = arr_time
        self.priority = priority #LP or HP
        self.service_time_left = service_time_left
        self.server_serving = server_serving

class Server:
    def __init__(self, id, status,client_served=None):
        self.id = id 
        self.status = status #int: 0 if idle, 1 if busy
        self.client_served = client_served

def generate_arrival_times(lambda_arrival): #function to generate EXPONENTIAL interarrival times 
    return  np.random.exponential(lambda_arrival)

def generate_exp_service(lambda_service): #function to generate EXPONENTIAL (with mean = 1 ) service times 
    return  np.random.exponential(1/lambda_service)

def generate_det_service(lambda_service): #function to generate DETERMINISTIC service times 
    return  lambda_service

def generate_hyperexp_service(case_mean_std, priority): #function to generate HYPEREXPONENTIAL service times 
    ''' parameters below are found solving the linear system associated to our target mean and std'''
    p = .5

    if case_mean_std == "a":
        lambda1=1/6
        lambda2=1/8
        u = np.random.uniform(0,1)
        if u <= p:
            x = np.random.exponential(1/lambda1)
        else: 
            x = np.random.exponential(1/lambda2)

    elif case_mean_std == "b":
        if priority == "HP":
            mean = 1/2
            std = 5
            lambda1, lambda2 = symbols('lambda1 lambda2')
            eq1 = Eq(p/lambda1 + (1-p)/lambda2, mean)
            eq2 = Eq(2*p/(lambda1**2) + 2*(1-p)/(lambda2**2)-mean, std**2)
            solutions = solve((eq1,eq2), (lambda1, lambda2))
            l1 = solutions[0][0]
            l2 = solutions[0][1]
            u = np.random.uniform(0,1)
            if u <= p:
                x = np.random.exponential(1/l2)
            else: 
                x = np.random.exponential(1/l2)
        
        if priority == "LP":
            mean = 3/2
            std = 15
            lambda1, lambda2 = symbols('lambda1 lambda2')
            eq1 = Eq(p/lambda1 + (1-p)/lambda2, mean)
            eq2 = Eq(2*p/(lambda1**2) + 2*(1-p)/(lambda2**2)-mean, std**2)
            solutions = solve((eq1,eq2), (lambda1, lambda2))
            l1 = solutions[0][0]
            l2 = solutions[0][1]
            u = np.random.uniform(0,1)
            if u <= p:
                x = np.random.exponential(1/l2)
            else: 
                x = np.random.exponential(1/l2)
                
    return x

def generate_service_time(s_gen_type, case_mean_std, priority): #function that recall all the service times functions, based on the type considered

    if s_gen_type == "exp":
        if case_mean_std == "a":
            lambda_service = 1
        elif case_mean_std == "b":
            if priority == "LP":
                lambda_service = 3/2
            elif priority == "HP":
                lambda_service = 1/2
        serviceTime = generate_exp_service(lambda_service)
    elif s_gen_type == "det":
        if case_mean_std == "a":
            lambda_service = 1
        elif case_mean_std == "b":
            if priority == "LP":
                lambda_service = 3/2
            elif priority == "HP":
                lambda_service = 1/2
        serviceTime = generate_det_service(lambda_service)
    elif s_gen_type == "hyperexp":
        serviceTime = generate_hyperexp_service(case_mean_std, priority)

    return serviceTime
        
       
def arrival(time,FES, clientsList,users,s_gen_type,lambda_arrival_both,N,case_mean_std,server): #function to manage events = 'arrival'
    
    arr_time =  np.random.exponential(1/lambda_arrival_both)
    priority = np.random.choice(["LP","HP"])
    serviceTime = generate_service_time(s_gen_type,case_mean_std, priority)

    if len(clientsList)<=N:
        c = Client(time,priority,service_time_left=serviceTime) 
        FES.append((time+arr_time,"arrival",priority)) #update FES
        users = users + 1 #update the user counter
        clientsList.append(c)  #add the created client in the queue
    if users == 1: #if I have only 1 client in the queue I need also to generate its service time
        server.client_served = c
        c.server_serving = server
        server.status = 1
        FES.append((time+serviceTime, "departure", priority))

    return users

def HP_arrival(servers, users, time, FES, clientsList, s_gen_type, case_mean_std, lambda_arrival_both, N): #to manage arrivals when all the servers are busy and the client is HP
        # choose randomly one server to make it idle in order to serve the HP client
        chosen_server = np.random.choice(servers)
        #priority = np.random.choice(["LP","HP"])
        priority = "HP"
        arrivalTime = generate_arrival_times(lambda_arrival_both)
        serviceTime = generate_service_time(s_gen_type,case_mean_std,priority)
        if len(clientsList) <= N:
            FES.append((time+arrivalTime, "arrival", priority))
            users = users + 1
            c = Client(arrivalTime,priority,service_time_left=serviceTime)
            clientsList.append(c)

        #client stopped is put again in the queue
        stopped_client = chosen_server.client_served
        stopped_client.server_serving = None
        stopped_client.arr_time = time
        if users == 1:
            chosen_server.client_served = stopped_client
            stopped_client.server_serving = chosen_server
            chosen_server.status = 1
            FES.append((time+serviceTime, "departure", priority))
        if len(clientsList)<=N:
            users = users + 1
            FES.insert(0,(time+stopped_client.service_time_left, "arrival", stopped_client.priority))

        return users

def departure(all_delays,delays_LP,delays_HP,time,FES,clientsList,users,s_gen_type,case_mean_std,priority): #function to manage events = 'departure'
    
    client_popped = clientsList.pop(0) #since the event is departure I delete the client from the queue
    priority = client_popped.priority
    #update delays lists
    if priority == "LP":
        delays_LP.append(time-client_popped.arr_time)
    elif priority == "HP":
        delays_HP.append(time-client_popped.arr_time)
    all_delays.append(time-client_popped.arr_time)

    server = client_popped.server_serving
    if server != None:
        server.status = 0
    users = users - 1 #and decrement the counter
    
    if users > 0: #I can have departures only if there is at least 1 client
        serviceTime = generate_service_time(s_gen_type,case_mean_std,priority)
        FES.append((time + serviceTime, "departure",priority)) #update FES

    return users,all_delays,delays_LP, delays_HP

def cumulative_mean(delays): #function used to compute the cumulative mean of the clients delay times list 
    cum_sum = np.cumsum(delays,axis=0)
    for i in range(cum_sum.shape[0]):
        if i == 0:
            continue
        cum_sum[i] = cum_sum[i]/(i+1)
    return list(cum_sum)


def plot_delays(all_delays,delays_LP,delays_HP,title,filefolder,filename):

    plt.plot(all_delays,label = 'All delays')
    plt.plot(delays_LP,label = 'LP delays')
    plt.plot(delays_HP,label = 'HP delays')
    plt.title(title)
    plt.xlabel('#customers')
    plt.ylabel('Delay')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()
  
def main():

    lambda_arrivals = [0.2, 0.4, 0.8, 1.4, 2.0, 2.4,2.8]
    s_gen_types = ["exp","det","hyperexp"]
    cases = ["a","b"]

    for s_gen_type in s_gen_types:
        for lambda_arrival_both in lambda_arrivals:
            for case_mean_std in cases:
                print(f'Using distribution for service times:{s_gen_type}')
                print(f'Lambda for arrivals : {lambda_arrival_both}')
                print(f'Case for mean and std or service times: {case_mean_std}')

                users = 0 #counter of client in the queue
                sim_time = 0.0 #current time, it is updated whenever we have an event

                
                #just for debugging purposes
                arr = 0
                dep = 0

                if s_gen_type == "hyperexp":
                    MAXTIME = 1000 #due to high computational needs
                else:
                    MAXTIME = 10000 #maximun running simulation time

                FES = list() #future event set 
                N = 1000 #waiting line size
                clientsList = list() #queue of clients
                all_delays = list() #list containing client delay times 
                delays_LP = list() #list containing client delay times for LP clients
                delays_HP = list() #list containing client delay times for HP clients

                k = 2 #number of servers
                server_list = list()
                for i in range(k):
                    server_list.append(Server(id = i, status = 0))

                #before entering the main loop I have to insert in the FES the first event
                FES.append((sim_time,"arrival","LP")) #recall: time is initialized = 0

                #simulation loop

                while sim_time < MAXTIME:
                    FES.sort(key=lambda y: y[2]) #FES sorted by event priority HP<LP true 
                    #print(FES)

                    #I extract the first (ordered by priority type) event 
                    #current time is updated to the scheduling time of the event
                    time, event, priority = FES.pop(0)

                    if time < sim_time:  
                        time = sim_time
                    sim_time = time

                    if event == "arrival": #based on the type of the event extracted I recall one of the function above
                        for i in range(k-1):
                            if server_list[i].status == 0:
                                arr = arr+1
                                users = arrival(sim_time,FES, clientsList,users,s_gen_type,lambda_arrival_both, N,case_mean_std,server_list[i])
                            elif server_list[i+1].status == 0:
                                arr = arr+1
                                users = arrival(sim_time,FES, clientsList,users,s_gen_type,lambda_arrival_both, N,case_mean_std,server_list[i+1])
                            elif priority == "HP":
                                if server_list[i].client_served.priority == "HP" and server_list[i+1].client_served.priority == "HP":
                                    users = arrival(sim_time,FES, clientsList,users,s_gen_type,lambda_arrival_both, N,case_mean_std,server_list[i])
                                else:
                                    users = HP_arrival(server_list,users,sim_time,FES,clientsList,s_gen_type,case_mean_std,lambda_arrival_both,N)
                                arr = arr+1
                    elif event == "departure":
                        dep = dep+1
                        users, all_delays, delays_LP, delays_HP = departure(all_delays,delays_LP,delays_HP,sim_time,FES, clientsList,users,s_gen_type,case_mean_std,priority) 

                print(f'# Arrivals: {arr}, # Departures:{dep}') #just for debugging purposes

                
                plot_delays(
                        #compute cumulative mean of delays, in order to understand their trends
                        all_delays=cumulative_mean(all_delays),
                        delays_LP=cumulative_mean(delays_LP),
                        delays_HP=cumulative_mean(delays_HP),
                        title=f'Delays,service time generation: {s_gen_type}, case: {case_mean_std}, lambda_arrival: {lambda_arrival_both}',
                        filefolder=f'delays_plots/{s_gen_type}',
                        filename=f'arr{lambda_arrival_both}_{s_gen_type}_{case_mean_std}.png'
                        )
                
                
if __name__=='__main__':
    main()
