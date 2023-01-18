import random 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats

def count_nodes(node, node_count): 
  infected = []
  node_count += 1 #increment the node_count
  if node["n_child"] == 0: 
    #leaf node
    infected.append((node["time"], node_count))
  else: 
    #Non leaf node
    for child in node["children"]: 
      infected.extend(count_nodes(child, node_count))
  return infected

def count_death_nodes(node, death_count): 
  death = []
  if np.random.uniform() < 0.02: #2% probability of being a death node
      death_count += 1 
  if node["n_child"] == 0: 
    #lead node
    death.append((node["time"], death_count))
  else: 
    #non-leaf node
    for child in node["children"]: 
      death.extend(count_death_nodes(child, death_count))
  return death

def generate_tree(ancestor, max_time, h): 
    num_children = ancestor["n_child"]
    #print(f"Generating children for ancestor with time {ancestor['time']}")
    for i in range(num_children): 
        child_time = ancestor["time"] + h()
        n_child = np.random.poisson(2)
        child = {"time": child_time, "n_child": n_child, "children": []}
        if child["time"] >= max_time: 
            return ancestor
        else: 
            ancestor["children"].append(generate_tree(child, max_time, h))
    return ancestor

def plot_totals(tree_list): 
  infected_list = list()
  death_list = list()
  inf_count = 0 
  death_count = 0 
  for tree_ in tree_list: 
      infected_list.extend(count_nodes(tree_, inf_count))
      death_list.extend(count_death_nodes(tree_, death_count))

  #SORT BY TIME 
  infected_list.sort(key=lambda x: x[0])
  death_list.sort(key=lambda x: x[0])

  #CUMULATIVE SUM ON THE SORTED LIST 
  result_infected = []
  sum = 0
  for time, number in infected_list:
      sum += number
      result_infected.append((time, sum))

  result_death = []
  sum = 0
  for time, number in death_list:
      sum += number
      result_death.append((time, sum))

  x, y = zip(*result_infected)
  x1, y1 = zip(*result_death)
  plt.plot(x, y, label="Infected")
  plt.plot(x1, y1, label="Death")
  plt.xlabel("Time")
  plt.ylabel("Count")
  plt.grid()
  plt.legend()
  plt.show()


def death_over_year(trees):
  num_deaths = 0
  for tree in trees:
    if tree["is_dead"]:
      num_deaths += 1
    num_deaths += death_over_year(tree["children"])
  return num_deaths
