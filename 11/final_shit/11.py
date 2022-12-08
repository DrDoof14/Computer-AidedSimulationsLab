import random
import numpy as np

random.seed(66)
np.random.seed(66)


# Class of server to serve each client
class Server:
    def __init__(self, server_name, server_status=0):
        self.server_name = server_name
        self.server_status = server_status

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.arrival_time_list = []
        self.departure_time = 0

    def empty(self):
        return len(self.elements) == 0

    # Putting each client in the list to be served and set an arrival time for that by piosson distribution
    def put(self, item, priority, arrival_time):
        self.elements.append([priority, item])
        inter_arrival = np.random.poisson(2)
        self.arrival_time_list.append(arrival_time + inter_arrival)
        return arrival_time + inter_arrival


    def get(self):
        # Get the highest priority element to serve it first
        highest_priority_element = max(self.elements, key=lambda x: x[0])
        highest_priority_index = self.elements.index(highest_priority_element)
        
        if self.departure_time == 0:
            self.departure_time = self.arrival_time_list[highest_priority_index] + service_time
        else:
            self.departure_time = self.departure_time + service_time
        
        # self.departure_time = self.arrival_time_list[highest_priority_index] + service_time
        
        arr_time = self.arrival_time_list[highest_priority_index]
        
        # Remove the highest priority element
        self.elements.remove(highest_priority_element)
        del self.arrival_time_list[highest_priority_index]
        
        # Return the highest priority element
        return highest_priority_element, arr_time, self.departure_time

import math
def hyper_expo() -> float:


    p = .5
    l1 = 1/6
    l2 = 1/8
    u = random.random()
    if u <= p:
        expec = l1
    else:
        expec = l2
    service = np.random.exponential(1/expec)
    
    return service




# Generating desired number of servers
S1 = Server(server_name='S1')
S2 = Server(server_name='S2')
server_list = ['S1', 'S2']
server_number = 2

# Initiate priority class
pq = PriorityQueue()

# Initiate params of simulation
service_time_distribution = ['deterministic', 'expo', 'hyperexpo']
lambda_list = [0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8]

arrive_time = 0
counter = 0
leaving_time_s1 = 0
leaving_time_s2 = 0
queue_length = 0
queue_length_max = 100

delay_LP = []
delay_HP = []
delay_total = []

ave_delay_LP = []
ave_delay_HP = []
ave_delay_total = []
avg_delay_dict={}

#iterating through different values for lambda and different distributions
for la in lambda_list:
    for dist in [service_time_distribution[1]]:
            # Start simulation
            while arrive_time < 1000 and queue_length < queue_length_max: #N=1000

                # Generating service time with the desired distribution
                if dist == 'expo': #exponential
                    service_time = np.random.exponential(1)

                elif dist=='deterministic':
                    service_time = 1
                else: #hyper-exponential
                    service_time = hyper_expo()


                for _ in range(server_number):
                    # arrive_time += 1
                    counter += 1
                    priority = random.randint(0, 1)
                    server_name = server_list[random.randint(0, 1)]
                    arrive_time = pq.put('client' + str(counter), priority, arrive_time)

                    queue_length +=1

                    # check if the servers can be in the serving condition
                    if leaving_time_s1 > arrive_time:
                        S1.server_status = 1
                    else:
                        S1.server_status = 0

                    if leaving_time_s2 > arrive_time:
                        S2.server_status = 1
                    else:
                        S2.server_status = 0

                    # serving clients
                    if S1.server_status == 0 and not pq.empty(): # cheking is the server is empty or not
                        a, get_in_time_s1, leaving_time_s1 = pq.get()
                        S1.server_status = 1

                        if a[0] == 1:
                            p = "HP"
                            delay_HP.append(leaving_time_s1-get_in_time_s1) 
                            ave_delay_HP.append(sum(delay_HP)/len(delay_HP))

                            delay_total.append(leaving_time_s1-get_in_time_s1)
                            ave_delay_total.append(sum(delay_total)/len(delay_total)) #computing the average delay
                        else:
                            p = "LP"
                            delay_LP.append(leaving_time_s1-get_in_time_s1) 
                            ave_delay_LP.append(sum(delay_LP)/len(delay_LP)) #computing the average delay

                            delay_total.append(leaving_time_s1-get_in_time_s1)
                            ave_delay_total.append(sum(delay_total)/len(delay_total))

                        print(f'{a[1]} as {p} arrived at {get_in_time_s1} and departed at {leaving_time_s1} from server {S1.server_name} with service_time {service_time}')

                        queue_length -=1

                    if S2.server_status == 0 and not pq.empty():
                        a, get_in_time_s2, leaving_time_s2 = pq.get()
                        S2.server_status = 1

                        if a[0] == 1:
                            p = "HP"
                            delay_HP.append(leaving_time_s2-get_in_time_s2)  
                            ave_delay_HP.append(sum(delay_HP)/len(delay_HP))

                            delay_total.append(leaving_time_s2-get_in_time_s2)
                            ave_delay_total.append(sum(delay_total)/len(delay_total))
                        else:
                            p = "LP"
                            delay_LP.append(leaving_time_s2-get_in_time_s2) 
                            ave_delay_LP.append(sum(delay_LP)/len(delay_LP)) 

                            delay_total.append(leaving_time_s2-get_in_time_s2)
                            ave_delay_total.append(sum(delay_total)/len(delay_total))

                        print(f'{a[1]} as {p} arrived at {get_in_time_s2} and departed at {leaving_time_s2} from server {S2.server_name} with service_time {service_time}')

                        queue_length -=1
