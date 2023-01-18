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
    
    def h_t_(type_distrib, rho = 1):
        """
            To calculate h(t) according to the two types of distribution
            Use rho have more possibility to sample larger intervals of time
        """
        if type_distrib == "uniform":
            return np.random.uniform(0,20/rho)
        if type_distrib == "exp":
            lambda_ = rho/10
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

    def hw_process(ancestor, type_distrib, m, time_horizon, death_rate):
        """
            To simulate the hawkes process starting from a single ancestor
        """
        global infected_list 
        global dead_list
        global dead
        global infected
        global rho_t
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

        # if the node is still alive, then make it infect other nodes (generate children)
        if ancestor.alive:
            tao = Utils.h_t_(type_distrib)
            t = ancestor.time_born + tao
            for _ in range(ancestor.children):
                # if we are in the non-pharmaceutical interventio phase and there is at least 1 dead node
                if t > 20 and dead > 0:
                    # adjust rho as the square root of the fracion of dead over infected nodes
                    rho = np.sqrt(dead/infected)
                    # take trace of the value of rho at each time
                    if ancestor.time_born in rho_t.keys():
                        rho_t[ancestor.time_born].append(rho)
                    else:
                        rho_t[ancestor.time_born] = [rho]

                    tao = Utils.h_t_(type_distrib, rho)
                    t = ancestor.time_born + tao
                    Y = np.random.poisson(m*rho)

                else:
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

    def rho_and_costs(rho_t):
        """
            Calculate the costs as the cumulative sum of rho^2
            return rho_t -> List(Tuple(time, value)), costs -> List(Tuple(time, cumulative value))
        """
        costs = []
        tmp = np.mean(rho_t[list(rho_t.keys())[0]])**2
        for key in rho_t.keys():
            rho_t[key] = np.mean(rho_t[key])
            costs.append((key, tmp))
            tmp += np.mean(rho_t[key])**2
        return rho_t, costs

    def run_times(lista, final_diz):
        """
            To insert the value of a list of type List(Tuple(time, value)) in a dictionary of type Dict(day, List(value))
            Return the updated dictionary
        """
        for el in lista:
            if int(np.floor(el[0])) in final_diz.keys():
                final_diz[int(np.floor(el[0]))].append(el[1])
            else:
                final_diz[int(np.floor(el[0]))] = [el[1]]
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
global rho_t

def main():
    np.random.seed(300751)
    m = 2
    death_rate = 0.02
    time_horizon = 365
    type_distrib = "exp" # eventually o be changed in "uniform"
    RUNS = 10
    final_diz_inf = {}
    final_diz_dead = {}
    final_diz_costi = {}

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
        global rho_t
        rho_t = {}

        # Generate the ancestors
        X = Utils.ancestor_generation(m)

        processed_nodes = []

        # Start the Hawkes process and take trace of the processed nodes
        for ancestor in X:
            processed_nodes.append(Utils.hw_process(ancestor, type_distrib, m, time_horizon, death_rate))

        # Take trace of the infected and dead nodes of this run
        infected_list = Utils.cumulative_sum_dict(infected_list)
        dead_list = Utils.cumulative_sum_dict(dead_list)

        # Sort the information about rho over time
        rho_t = OrderedDict(sorted(rho_t.items(), key=lambda x: x[0]))
        # Calculate the costs
        rho_t, costi = Utils.rho_and_costs(rho_t)

        # Save the day-by-day information of dead and infected nodes and costs for this run
        final_diz_inf = Utils.run_times(infected_list, final_diz_inf)
        final_diz_dead = Utils.run_times(dead_list, final_diz_dead)
        final_diz_costi = Utils.run_times(costi, final_diz_costi)

    # Fill if there are missing parts because of 0 values
    for i in range(time_horizon):
        if i not in final_diz_dead:
            final_diz_dead[i] = 0
        if i<20:
            final_diz_costi[i] = 0

    # Sort the day-by-day dictionary of information on infected and dead nodes and costs of all the runs by day
    final_diz_inf = OrderedDict(sorted(final_diz_inf.items(), key=lambda x: x[0]))
    final_diz_dead = OrderedDict(sorted(final_diz_dead.items(), key=lambda x: x[0]))
    final_diz_costi = OrderedDict(sorted(final_diz_costi.items(), key=lambda x: x[0]))
    
    # Calculate the confidence interval for dead and infected nodes and costs
    lb_list_inf, ub_list_inf, mean_list_inf = Utils.conf_int(final_diz_inf, RUNS)
    lb_list_dead, ub_list_dead, mean_list_dead = Utils.conf_int(final_diz_dead, RUNS)
    lb_list_costi, ub_list_costi, mean_list_costi = Utils.conf_int(final_diz_costi, RUNS)

    # plot the costs
    plt.plot(mean_list_costi)
    plt.fill_between([x for x in range(len(mean_list_costi))], mean_list_costi - lb_list_costi, mean_list_costi + ub_list_costi, alpha = 0.3)
    plt.legend(labels = ["Costs", "CI - Costs"], loc = "upper left")
    plt.title(f"Costs for non pharmaceutical intervention - h(t): {type_distrib}")
    plt.xlabel("Time")
    plt.ylabel("Costs")
    plt.savefig(f"plots/part_2_cost_type_{type_distrib}.png")
    plt.show()
    plt.close()

    # plot infected and dead nodes over time
    plt.plot(mean_list_inf)
    plt.fill_between([x for x in range(len(mean_list_inf))], mean_list_inf - lb_list_inf, mean_list_inf + ub_list_inf, alpha = 0.3)
    plt.plot(mean_list_dead)
    plt.fill_between([x for x in range(len(mean_list_dead))], mean_list_dead - lb_list_dead, mean_list_dead + ub_list_dead, alpha = 0.3)
    plt.legend(labels = ["Infected", "CI - Infected", "Dead", "CI - Dead"], loc = "upper left")
    plt.title(f"Infected and Dead for non pharmaceutical intervention - h(t): {type_distrib}")
    plt.xlabel("Time")
    plt.ylabel("Infected & Dead nodes")
    plt.savefig(f"plots/part_2_infdead_type_{type_distrib}.png")
    plt.show()
    plt.close()

    

    
    

if __name__ == "__main__":
    main()