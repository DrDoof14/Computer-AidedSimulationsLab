import random


class Client:
    def __init__(self, name, arrival_time, departure_time, Server):
        self.name = name
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.Server = Server


class Server:
    def __init__(self, server_name):
        self.server_status = 0
        self.server_name = server_name


def kir(FES, cus_list, current_cliente):
    st = 0

    for i in range(len(FES)):
        print('inside kir')
        if FES[i][1] == 'departure' and FES[i][2] == 'LP':
            tmp = cus_list[i - 1]
            st = tmp.departure_time - tmp.arrival_time
            break
    current_cliente.departure_time = st+current_cliente.arrival_time
    FES.append([current_cliente.departure_time, 'departure', 'HP'])
    FES=sorted(FES)
    for j in range(len(FES)):
        if FES[j][1] == 'departure' and FES[j][2] == 'LP':
            FES[j][0] += st
    FES=sorted(FES)
    return FES, current_cliente


def arrival(time, FES, queue, average_arrival_time, average_service_time, users, customer, customer_list,
            server_name, Priority):
    inter_arrival = random.expovariate(1.0 / average_arrival_time)
    flg = random.randint(0, 1)
    if flg == 0:
        FES.append([time + inter_arrival, 'arrival', 'LP'])
        users += 1
        x = 'client' + str(customer)
        client = Client(x, time, 0, server_name)
        customer_list.append(client)
        queue.append(client)
    else:
        FES.append([time + inter_arrival, 'arrival', 'HP'])
        users += 1
        x = 'client' + str(customer)
        client = Client(x, time, 0, server_name)
        customer_list.append(client)
        FES, client = kir(FES, customer_list, client)
        FES.sort()
        queue.append(client)

    # managing the event
    # users += 1
    # x = 'client' + str(customer)
    # client = Client(x, time, 0, server_name)
    # customer_list.append(client)
    # queue.append(client)
    print('FES ', FES)
    input()
    customer += 1

    # recording client id and put it in the list
    print(f'{server_name} : {client.name} arrived at {client.arrival_time}')
    # start the service in case the server is idle
    if users == 1:
        service_time = random.expovariate(1.0 / average_service_time)
        FES.append([time + service_time, 'departure', Priority])
        waiting_time = time + service_time
    return users, customer, customer_list


def departure(time, FES, queue, average_arrival_time, average_service_time, users, customer, customer_list,
              server_name, Priority):
    # manipulating the list of clients to get FIFO orientation
    queue.reverse()
    client = queue.pop()
    queue.reverse()
    users -= 1
    delay = time - client.arrival_time

    print(f'{client.name} departed at {time}')

    # checking the number of clients in line
    if users > 0:
        service_time = random.expovariate(1.0 / average_service_time)
        FES.append([time + service_time, 'departure', Priority])
    return delay, users


def main():
    S1 = Server(server_name='S1')
    # S2 = Server(server_name='S2')
    FES, queue, customer_list = [], [], []
    customer, users, time, target = 0, 0, 0, 0
    FES.append([0, 'arrival', 'LP'])
    # customer_list.append()
    avg_arrival_time = 2
    average_service_time = 4
    io = 0
    while io < 100:
        io += 1
        FES = sorted(FES)
        (time, event_type, Priority) = FES[target]
        if event_type == 'arrival':
            users, customer, customer_list = arrival(time, FES, queue, avg_arrival_time, average_service_time, users,
                                                     customer,
                                                     customer_list, S1.server_name, Priority)
        elif event_type == 'departure':
            tmp_delay, users = departure(time, FES, queue, avg_arrival_time, average_service_time, users, customer,
                                         customer_list, S1.server_name, Priority)
            # delay.append(tmp_delay)
            # cumulative_delay.append(sum(delay) / len(delay))

        target += 1


if __name__ == "__main__":
    main()
