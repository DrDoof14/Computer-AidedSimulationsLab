import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os

np.random.seed(4899)

class Client:   #class of the object client
    def __init__(self, arr_time, type):
        self.arr_time = arr_time
        self.type = type

def generate_arrival_times(ARRIVAL): #function to generate EXPONENTIAL interarrival times 
    return  np.random.exponential(ARRIVAL)

def generate_exp_service(SERVICE): #function to generate EXPONENTIAL (with mean = 1 ) service times 
    return  np.random.exponential(SERVICE)

def generate_det_service(): #function to generate DETERMINISTIC service times 
    return  1

def generate_hyperexp_service(mean = 1, std= 10): #function to generate HYPEREXPONENTIAL service times 
    '''parameters below are found solving the linear system associated to our target mean and std
        see more on report'''
    p = .5
    lambda1=1/6
    lambda2=1/8
    u = np.random.uniform(0,1)
    if u <= p:
        x = np.random.exponential(1/lambda1)
    else: 
        x = np.random.exponential(1/lambda2)
    return x
       
def arrival(time,FES, clientsList,users,s_gen_type,u): #function to manage events = 'arrival'
    if s_gen_type == 'exp':
        SERVICE = 1
        ARRIVAL = SERVICE/u
        serviceTime = generate_exp_service(SERVICE)
    elif s_gen_type == 'det':
        serviceTime = generate_det_service()
        SERVICE = serviceTime
        ARRIVAL = SERVICE/u
    elif s_gen_type == 'hyperexp':
        serviceTime = generate_hyperexp_service()
        SERVICE = serviceTime
        ARRIVAL = SERVICE/u
    arr_time =  np.random.exponential(ARRIVAL)
    c = Client(time, 'arrival') 
    FES.append((time+arr_time,'arrival')) #update FES
    users = users + 1 #update the user counter
    clientsList.append(c)  #add the created client in the queue
    if users == 1: #if I have only 1 client in the queue I need also to generate its service time
        FES.append((time+serviceTime, 'departure'))
    return users

def departure(delays,time,FES,clientsList,users,s_gen_type): #function to manage events = 'departure'
    users = users - 1 #and decrement the counter
    client_popped = clientsList.pop(0) #since the event is departure I delete the client from the queue
    delays.append(time-client_popped.arr_time)
    if users > 0: #I can have departures only if there is at least 1 client
        if s_gen_type == 'exp':
            SERVICE = 1
            serviceTime = generate_exp_service(SERVICE)
        elif s_gen_type == 'det':
            serviceTime = generate_det_service()
            SERVICE = serviceTime
        elif s_gen_type == 'hyperexp':
            serviceTime = generate_hyperexp_service()
            SERVICE = serviceTime
        FES.append((time + serviceTime, "departure")) #update FES
    return users,delays

def cumulative_mean(delays): #function used to compute the cumulative mean of the clients delay times list 
    cum_sum = np.cumsum(delays,axis=0)
    for i in range(cum_sum.shape[0]):
        if i == 0:
            continue
        cum_sum[i] = cum_sum[i]/(i+1)
    return list(cum_sum)


def remove_transient(delays,u,thr): #function the detect and remove the transient 
    ''' thr per exp = 0.98
        thr per determionistica = 0.98
        thr per hyperxp = 0.9
    '''
    len_window = int(u*1000)  #experimentally found that it is a good approximation
    n_window = int(len(delays)/len_window)
    for i in range(n_window):
        if i*len_window >= len(delays):
            break
        else:
            window = delays[i*len_window: (i+1)*len_window] #select one window at time
            if window != []:
                min1,max1 = min(window),max(window) #save its min and max
                normalized = min1/max1 #if this ratio is too big means that we still have too much ups and dows, so the curve is not enough steady
            if normalized>thr: #so when too big remove the considered window ans start from the following one
                delays = delays[(i+1)*len_window:]
                break
    return delays,len_window*(i+1)

def find_n_batches(no_transient_delays,n, confidence=.95): #function finding the correct number of batches, using bacth means algorithm, to perform one last time batch means
    no_transient_delays = np.array(no_transient_delays)
    while True:
        batches = np.array_split(no_transient_delays, n)
        for batch in batches:
            x = np.mean(batch)
            ci = st.t.interval(confidence, len(batches)-1, x, np.std(batch))
            z = ci[1] - x #upper bound - mean = interval width
            #using 2z/x we are looking how much is the width of the interval with respect to its center
            if (2*z/x) > 0.15:
                n += 1
                break
            else: 
                print(f'Final number of batches:{n}')
                return n

def batch_means(no_transient_delays, correct_n, confidence=.95): #last iteration of the batch means, so compute confidence interval of the means
    means = list()
    ci_lower_bound= list()
    ci_upper_bound = list()
    no_transient_delays = np.array(no_transient_delays)
    batches = np.array_split(no_transient_delays, correct_n)
    for batch in batches:
        x = np.mean(batch)
        ci = st.t.interval(confidence, len(batches)-1, x, np.std(batch))
        ci_lower_bound.append(ci[0])
        ci_upper_bound.append(ci[1])
        means.append(x)
    return means, ci_lower_bound, ci_upper_bound
 
'''plotting functions'''

def plot_no_transient_delays(c_mean_delays,end_transient_idx,title,filefolder,filename):
    plt.plot(c_mean_delays, c = 'g')
    plt.axvline(end_transient_idx,c='r')
    plt.title(title)
    plt.xlabel('#customers')
    plt.ylabel('Delay')
    plt.grid()
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()

def plot_means_ci_delays(batch_means_delays,ci_lower_bound,ci_upper_bound,title,filefolder,filename):
    plt.plot(batch_means_delays, c = 'g')
    x = [x for x in range(len(batch_means_delays))]
    plt.fill_between(x,ci_lower_bound,ci_upper_bound,alpha=.2,color ='green',label=f'95% c.i.')
    plt.title(title)
    plt.xlabel('Batch id')
    plt.legend()
    plt.ylabel('Delay')
    plt.grid()
    plt.savefig(os.path.join(filefolder,filename))
    plt.close()
  
def main():
    us = [0.1,0.2,0.4,0.7,0.8,0.9,0.95,0.99]
    s_gen_types = ['exp','det','hyperexp']

    means_exp_no_t = list()
    lower_exp_no_t = list()
    upper_exp_no_t = list()

    means_det_no_t = list()
    lower_det_no_t = list()
    upper_det_no_t = list()

    means_hyperexp_no_t = list()
    lower_hyperexp_no_t = list()
    upper_hyperexp_no_t = list()

    for u in us:
        for s_gen_type in s_gen_types:
            print(f'Using u:{u}, distribution for service times:{s_gen_type}')
            users = 0 #counter of client in the queue
            time = 0.0 #current time, it is updated whenever we have an event

            
            #just for debugging purposes
            arr = 0
            dep = 0
        
            MAXTIME = 100000 #maximun running simulation time

            FES = list() #future event set 
            clientsList = list() #queue of clients
            delays = list() #list containing client delay times 

            #before entering the main loop I have to insert in the FES the first event
            FES.append((time,'arrival')) #recall: time is initialized = 0

            #simulation loop
    
            while time < MAXTIME:
                FES.sort(key=lambda y: y[0]) #FES sorted by occurence time of events
                event = FES.pop(0) #I extract the first (ordered by occurence time) event 
                time = event[0]  #current time is updated to the scheduling time of the event
                if event[1] == 'arrival': #based on the type of the event extracted I recall one of the function above
                    arr = arr+1
                    users = arrival(time,FES, clientsList,users,s_gen_type,u)
                elif event[1] == 'departure':
                    dep = dep+1
                    users, delays= departure(delays,time,FES, clientsList,users,s_gen_type) 

            print(f'# Arrivals: {arr}, # Departures:{dep}') #just for debugging purposes
          

            c_mean_delays= cumulative_mean(delays) #compute cumulative mean of the delay times list, in order to understand its trend
            
            '''
            in removing transient we have different threshold because data we analyze are different based on the distribution used to generate service times
            '''
            if s_gen_type == 'exp' or s_gen_type == 'det':
                thr = 0.98
            else :
                thr = 0.9
            no_transient_delays,end_transient_idx = remove_transient(c_mean_delays,u,thr) #remove transient from data, save the index of its end (just for plotting)

            plot_no_transient_delays(
                c_mean_delays=c_mean_delays,
                end_transient_idx=end_transient_idx,
                title=f'Avg delays, using u:{u},service time generation: {s_gen_type}',
                filefolder='delays_plots/',
                filename=f'delays_{u}_{s_gen_type}.png'
                )
            
           
            found_number_of_batches = find_n_batches(no_transient_delays,n=10) #compute the correct number of batches needed using the batch means method
            batch_means_delays, ci_lower_bound,ci_upper_bound = batch_means(no_transient_delays,found_number_of_batches) #using n_correct, compute ci and means
            plot_means_ci_delays(
                batch_means_delays,
                ci_lower_bound,
                ci_upper_bound,
                title=f'Batch means with u:{u}, service time generation: {s_gen_type}',
                filefolder='batch_means_ci_plots/',
                filename=f'bm_{u}_{s_gen_type}.png'
            )

            if s_gen_type=='exp':
                means_exp_no_t.append(np.mean(no_transient_delays))
                ci = st.t.interval(0.95,len(no_transient_delays)-1,np.mean(no_transient_delays),np.std(no_transient_delays))
                lower_exp_no_t.append(ci[0])
                upper_exp_no_t.append(ci[1])

            elif s_gen_type=='det':
                means_det_no_t.append(np.mean(no_transient_delays))
                ci = st.t.interval(0.95,len(no_transient_delays)-1,np.mean(no_transient_delays),np.std(no_transient_delays))
                lower_det_no_t.append(ci[0])
                upper_det_no_t.append(ci[1])
            
            elif s_gen_type=='hyperexp':
                means_hyperexp_no_t.append(np.mean(no_transient_delays))
                ci = st.t.interval(0.95,len(no_transient_delays)-1,np.mean(no_transient_delays),np.std(no_transient_delays))
                lower_hyperexp_no_t.append(ci[0])
                upper_hyperexp_no_t.append(ci[1])
            
            else:
                pass
    
    plt.plot(us,means_exp_no_t,label='exp')
    plt.plot(us,means_det_no_t,label='det')
    plt.plot(us,means_hyperexp_no_t,label='hyperexp')
    plt.fill_between(us,lower_exp_no_t,upper_exp_no_t,alpha=.2,label=f'95% c.i.')
    plt.fill_between(us,lower_det_no_t,upper_det_no_t,alpha=.2,label=f'95% c.i.')
    plt.fill_between(us,lower_hyperexp_no_t,upper_hyperexp_no_t,alpha=.2,label=f'95% c.i.')
    plt.title('Average delay (transient deleted) in function of U')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('U')
    plt.ylabel('Average')
    plt.savefig('avg_vs_u_no_t.png')
    plt.close()

     
if __name__=='__main__':
    main()
