import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import OrderedDict

class Node():
    """
        Class to represent all the nodes.
        time_born: time when the node is infected
        children: number of children of the node
        children_list: list of object Node containing all the children
        alive: boolean to flag if the node is alive or not
    """
    def __init__(self, time_born, children = 0, children_list = [], alive = True):
        self.time_born = time_born
        self.children = children
        self.children_list = children_list
        self.alive = alive

class Utils:
    def sigma():
        """
            To generate the time sigma at which the ancestors were infected
        """
        return np.random.exponential(1/20)
    
    def h_t_(type_distrib):
        """
            To calculate h(t) according to the two types of distribution
        """
        if type_distrib == "uniform":
            return np.random.uniform(0,20)
        if type_distrib == "exp":
            lambda_ = 1/10
            return np.random.exponential(1/lambda_)

    def ancestor_generation(m):
        """
            To generate and return all the ancestors of our simulation
        """
        X = []
        time_ancestor = 0
        while time_ancestor <= 10:
            time_ancestor += Utils.sigma()
            Y = np.random.poisson(m)
            X.append(Node(time_born=time_ancestor, children=Y))
        return X
    
    def hw_process(ancestor, type_distrib, m, time_horizon, death_rate):
        """
            To simulate the hawkes process starting from a single ancestor
        """
        global infected_list 
        global dead_list
        global dead
        global infected
        # increment the number of infected nodes
        infected += 1
        # take trace of at which time each node has been infected
        infected_list.append((ancestor.time_born, 1))

        # make the ancestor die with probability 2%
        u = np.random.uniform(0,1)
        if u < death_rate:
            ancestor.alive = False
            # increment the number of dead nodes
            dead += 1
            # take trace of at which time the node has died
            dead_list.append((ancestor.time_born, 1))
            return ancestor
            
        # if the node is still alive, then make it infect other nodes (generate children)
        if ancestor.alive:
            tao = Utils.h_t_(type_distrib)
            t = ancestor.time_born + tao
            for _ in range(ancestor.children):
                Y = np.random.poisson(m)
                child = Node(time_born=t, children = Y)
                # if the new node is over the time horizon, then return
                if t > time_horizon:
                    return ancestor
                # otherwise start another hawkes process recursively on rach child
                else:
                    ancestor.children_list.append(Utils.hw_process(child, type_distrib, m, time_horizon, death_rate))
        # return the processed ancestor
        return ancestor

    def cumulative_sum_dict(lista):
        """
            To calculate the cumulative sum of the value of a list of type List(Tuple(time, value))
        """
        lista.sort(key = lambda x: x[0])
        new_lista = []
        tmp = lista[0][1]
        for tupla in lista:
            new_lista.append((tupla[0], tmp))
            tmp = tmp+tupla[1]
        return new_lista

    def run_times(lista, final_diz):
        """
            To insert the value of a list of type List(Tuple(time, value)) in a dictionary of type Dict(day, List(value))
            Return the updated dictionary
        """
        for el in lista:
            if np.floor(el[0]) in final_diz.keys():
                final_diz[np.floor(el[0])].append(el[1])
            else:
                final_diz[np.floor(el[0])] = [el[1]]
        return final_diz


    def conf_int(final_dict, runs):
        """
            To calculate the 95% confidence interval for each day in the dicitionary Dict(day, List(value))
        """
        lb_list = []
        ub_list = []
        mean_list = []
        for lista in final_dict.values():
            lb_list.append(st.t.interval(confidence = 0.05, df = runs-1, loc = np.mean(lista), scale = np.std(lista))[0])
            ub_list.append(st.t.interval(confidence = 0.05, df = runs-1, loc = np.mean(lista), scale = np.std(lista))[1])
            mean_list.append(np.mean(lista))
        return np.array(lb_list), np.array(ub_list), np.array(mean_list)

        


global infected_list
global dead_list
global dead
global infected

def main():
    np.random.seed(300751)
    m = 2
    death_rate = 0.02
    time_horizon = 100
    RUNS = 10
    types = ["exp", "uniform"]

    for type_distrib in types:
        print(f"===== Type: {type_distrib} =====")
        final_diz_inf = {}
        final_diz_dead = {}

        for run in range(RUNS):
            print("Run n°", run)
            global infected_list
            infected_list = []
            global dead_list
            dead_list = []
            global dead
            dead = 0
            global infected
            infected = 0

            # Generate the ancestors
            X = Utils.ancestor_generation(m)
            
            processed_nodes = []

            # Start the Hawkes process and take trace of the processed nodes
            for ancestor in X:
                processed_nodes.append(Utils.hw_process(ancestor, type_distrib, m, time_horizon, death_rate))
            
            # Take trace of the infected and dead nodes of this run
            infected_list = Utils.cumulative_sum_dict(infected_list)
            dead_list = Utils.cumulative_sum_dict(dead_list)

            # Save the day-by-day information of dead and infected nodes for this run
            final_diz_inf = Utils.run_times(infected_list, final_diz_inf)
            final_diz_dead = Utils.run_times(dead_list, final_diz_dead)

        # Sort the day-by-day dictionary of information on infected and dead nodes of all the runs by day
        final_diz_inf = OrderedDict(sorted(final_diz_inf.items(), key=lambda x: x[0]))
        final_diz_dead = OrderedDict(sorted(final_diz_dead.items(), key=lambda x: x[0]))
        
        # Calculate the confidence interval for dead and infected nodes
        lb_list_inf, ub_list_inf, mean_list_inf = Utils.conf_int(final_diz_inf, RUNS)
        lb_list_dead, ub_list_dead, mean_list_dead = Utils.conf_int(final_diz_dead, RUNS)

        # plot infected and dead nodes over time
        plt.plot(mean_list_inf)
        plt.fill_between([x for x in range(len(mean_list_inf))], mean_list_inf - lb_list_inf, mean_list_inf + ub_list_inf, alpha = 0.3)
        plt.plot(mean_list_dead)
        plt.fill_between([x for x in range(len(mean_list_dead))], mean_list_dead - lb_list_dead, mean_list_dead + ub_list_dead, alpha = 0.3)
        plt.legend(labels = ["Infected", "CI - Infected", "Dead", "CI - Dead"], loc = "upper left")
        plt.xlabel("Time")
        plt.ylabel("Infected & Dead nodes")
        plt.title(f"Infected and Dead - Type for h(t): {type_distrib}")
        plt.savefig(f"plots/part_1_type_{type_distrib}.png")
        plt.show()
        plt.close()
        




if __name__ == "__main__":
    main()