import random


class Server:
    def __init__(self, server_name, server_status=0):
        self.server_name = server_name
        self.server_status = server_status


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.arrival_time_list = []
        self.departure_time_list = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority, arrival_time):
        self.elements.append([priority, item])
        inter_arrival = random.expovariate(1.0 / 2)
        self.arrival_time_list.append(arrival_time + inter_arrival)
        self.departure_time_list.append(0)

    def get(self):
        # Get the highest priority element
        highest_priority_element = max(self.elements, key=lambda x: x[0])
        highest_priority_index = self.elements.index(highest_priority_element)
        if self.departure_time_list[highest_priority_index] == 0:
            self.departure_time_list = [self.arrival_time_list[highest_priority_index] + service_time for _ in
                                        self.departure_time_list]
        else:
            self.departure_time_list = [element + service_time for element in self.departure_time_list]
        dep_time = self.departure_time_list[highest_priority_index]
        arr_time = self.arrival_time_list[highest_priority_index]
        # Remove the highest priority element
        self.elements.remove(highest_priority_element)
        del self.departure_time_list[highest_priority_index]
        del self.arrival_time_list[highest_priority_index]
        # Return the highest priority element
        return highest_priority_element, arr_time, dep_time


S1 = Server(server_name='S1')
S2 = Server(server_name='S2')
pq = PriorityQueue()
service_time = 4
arrive_time = 0
counter = 0
c1 = 0
c2 = 0
for i in range(100):
    arrive_time += 1
    counter += 1
    priority = random.randint(0, 1)
    pq.put('client' + str(counter), priority, arrive_time)

    arrive_time += 1
    counter += 1
    priority = random.randint(0, 1)
    pq.put('client' + str(counter), priority, arrive_time)
    # arrive_time += 1
    print(pq.elements)
    print(pq.arrival_time_list)
    print('===========')
    if c1 > arrive_time:
        S1.server_status = 1
    else:
        S1.server_status = 0
    if c2 > arrive_time:
        S2.server_status = 1
    else:
        S2.server_status = 0
    if S1.server_status == 0:
        a, b, c1 = pq.get()
        print(S1.server_name, a, b, c1, sep=' - ')
    if S2.server_status == 0:
        a, b, c2 = pq.get()
        print(S2.server_name, a, b, c2, sep=' - ')
    input()

for i in range(5):
    pass
