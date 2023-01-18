import hawkesProcess 
import tree 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import simulate_epidemic
from scipy import stats
from scipy.stats import norm

random.seed(310199)

### FIRST PART ### 
MAX_TIME = 100


ancestors = hawkesProcess.poisson_process(hawkesProcess.sigma, 10)

tot_child = 0
ancestors_ = list()
for ancestor in ancestors: 
    (time, n_child) = ancestor 
    ancestors_.append({"time": time, "n_child": n_child, "children": []})
    tot_child += n_child

print("Number of children at first generation", tot_child)

### UNIFORM ###

tree_list = list()

for ancestor in ancestors_ : 
    tree_= tree.generate_tree(ancestor, MAX_TIME, hawkesProcess.uniform)
    tree_list.append(tree_)

print("-----UNIFORM-----")
tree.plot_totals(tree_list)

"""
## CONFIDENCE INTERVALS ##

# Run the simulation and plot
num_sims = 4
infected_means, infected_cis = tree.run_simulation(ancestor, 100, num_sims, hawkesProcess.uniform)
tree.plot_interval_counts(infected_means, infected_cis)
"""

### EXPONENTIAL ###

tree_list = list()

for ancestor in ancestors_ : 
    tree_= tree.generate_tree(ancestor, MAX_TIME, hawkesProcess.exponential)
    tree_list.append(tree_)

print("-----EXPONENTIAL-----")
tree.plot_totals(tree_list)

### EPIDEMIC ### 

NUM_DAYS = 365

death_sum = 0
cost_sum = 0
infected_sum = 0
death_sum2 = 0
cost_sum2 = 0
infected_sum2 = 0
death_sum3 = 0
cost_sum3 = 0
infected_sum3 = 0
death_sum4 = 0 
cost_sum4 = 0 
infected_sum4 = 0 

death_list = list()
death_list2 = list()
death_list3 = list()
death_list4 = list()

n_sims = 5
for i in range(n_sims):
    infected, death, cost, _, _, _, = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.constant)
    infected2, death2, cost2, _, _, _, = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.stairs)
    infected3, death3, cost3, _, _, _, = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.sigmoid)
    infected4, death4, cost4, _, _, _, = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.exp_rho)
    death_list.append(len(death))
    infected_sum += len(infected)
    death_sum += len(death)
    cost_sum += cost

    death_list2.append(len(death2))
    infected_sum2 += len(infected2)
    death_sum2 += len(death2)
    cost_sum2 += cost2

    death_list3.append(len(death3))
    infected_sum3 += len(infected3)
    death_sum3 += len(death3)
    cost_sum3 += cost3

    death_list4.append(len(death4))
    infected_sum4 += len(infected4)
    death_sum4 += len(death4)
    cost_sum4 += cost4

avg_death = death_sum/n_sims
avg_cost = cost_sum/n_sims
avg_inf = infected_sum/n_sims 

avg_death2 = death_sum2/n_sims
avg_cost2 = cost_sum2/n_sims
avg_inf2 = infected_sum2/n_sims 

avg_death3 = death_sum3/n_sims
avg_cost3 = cost_sum3/n_sims
avg_inf3 = infected_sum3/n_sims 

avg_death4 = death_sum4/n_sims
avg_cost4 = cost_sum4/n_sims
avg_inf4 = infected_sum4/n_sims 

print("\nCONSTANT")
print("cost: ", avg_cost, "€")
print("death: ", avg_death)  
print("infected", avg_inf)

print("\nSTAIRS")
print("cost: ", avg_cost2, "€")
print("death: ", avg_death2)  
print("infected", avg_inf2)

print("\nSIGMOID")
print("cost: ", avg_cost3, "€")
print("death: ", avg_death3)  
print("infected", avg_inf3)

print("\nEXPONENTIAL")
print("cost: ", avg_cost4, "€")
print("death: ", avg_death4)  
print("infected", avg_inf4)

#confidence intervals

death_array = death_list
conf_int = norm.interval(0.95, loc=np.mean(death_array), scale=stats.sem(death_array))
death_array2 = death_list2
conf_int2 = norm.interval(0.95, loc=np.mean(death_array2), scale=stats.sem(death_array2))
death_array3 = death_list3
conf_int3 = norm.interval(0.95, loc=np.mean(death_array3), scale=stats.sem(death_array3))
death_array4 = death_list4
conf_int4 = norm.interval(0.95, loc=np.mean(death_array4), scale=stats.sem(death_array4))

plt.plot(death_list,  label = "constant", color='skyblue')
plt.axhline(y = avg_death, color = 'firebrick', linestyle = '-', label="average deaths constant")
plt.ylabel("Number of deaths")
plt.xlabel("Number of iterations")
plt.ylim([12500, 27500])
plt.errorbar(1, np.mean(death_array), yerr=stats.sem(death_array), fmt='o')
plt.fill_between(x=range(len(death_list)), y1=conf_int[0], y2=conf_int[1], alpha=0.2)
plt.grid(True)
plt.legend()
plt.show()


plt.plot(death_list2,  label = "stair", color='skyblue')
plt.axhline(y = avg_death2, color = 'firebrick', linestyle = '-', label="average deaths stairs")
plt.ylabel("Number of deaths")
plt.xlabel("Number of iterations")
plt.ylim([12500, 27500])
plt.errorbar(1, np.mean(death_array2), yerr=stats.sem(death_array2), fmt='o')
plt.fill_between(x=range(len(death_list2)), y1=conf_int2[0], y2=conf_int2[1], alpha=0.2)
plt.grid(True)
plt.legend()
plt.show()


plt.plot(death_list3,  label = "sigmoid", color='skyblue')
plt.axhline(y = avg_death3, color = 'firebrick', linestyle = '-', label="average deaths sigmoid")
plt.ylabel("Number of deaths")
plt.xlabel("Number of iterations")
plt.ylim([12500, 27500])
plt.errorbar(1, np.mean(death_array3), yerr=stats.sem(death_array3), fmt='o')
plt.fill_between(x=range(len(death_list3)), y1=conf_int3[0], y2=conf_int3[1], alpha=0.2)
plt.grid(True)
plt.legend()
plt.show()

plt.plot(death_list4,  label = "exponential", color='skyblue')
plt.axhline(y = avg_death4, color = 'firebrick', linestyle = '-', label="average deaths exponential rho")
plt.ylabel("Number of deaths")
plt.xlabel("Number of iterations")
plt.ylim([12500, 27500])
plt.errorbar(1, np.mean(death_array4), yerr=stats.sem(death_array4), fmt='o')
plt.fill_between(x=range(len(death_list4)), y1=conf_int4[0], y2=conf_int4[1], alpha=0.2)
plt.grid(True)
plt.legend()
plt.show()


#Plot of h during time 
_, _, _, h_values, rho_values, rate_value = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.constant)
_, _, _, h_values2, rho_values2, rate_value2 = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.stairs)
_, _, _, h_values3, rho_values3, rate_value3 = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.sigmoid)
_, _, _, h_values4, rho_values4, rate_value4 = simulate_epidemic.simulate_epidemic(NUM_DAYS, simulate_epidemic.exp_rho)


plt.plot(h_values.keys(), h_values.values(), color = "skyblue", label="constant")
plt.plot(h_values2.keys(), h_values2.values(), color = "cadetblue", label="stairs")
plt.plot(h_values3.keys(), h_values3.values(), color = "firebrick", label="sigmoid")
plt.plot(h_values4.keys(), h_values4.values(), color = "coral", label="exponential")
plt.grid()
plt.xlabel("Days")
plt.ylabel("Infections last 20 days (h)")
plt.legend()
plt.show()

#Plot of rho functions starting from day 20 
plt.plot(rho_values.keys(), rho_values.values(), color = "skyblue", label="constant")
plt.plot(rho_values2.keys(), rho_values2.values(), color = "cadetblue", label="stairs")
plt.plot(rho_values3.keys(), rho_values3.values(), color = "firebrick", label="sigmoid")
plt.plot(rho_values4.keys(), rho_values4.values(), color = "coral", label="exponential")
plt.grid()
plt.xlabel("Days")
plt.xlim(19, 365)
plt.ylabel("Value of rho")
plt.legend()
plt.show()

#Plot the rate 
plt.plot(rate_value.keys(), rate_value.values(), color = "skyblue", label="constant")
plt.plot(rate_value2.keys(), rate_value2.values(), color = "cadetblue", label="stairs")
plt.plot(rate_value3.keys(), rate_value3.values(), color = "firebrick", label="sigmoid")
plt.plot(rate_value4.keys(), rate_value4.values(), color = "coral", label="exponential")
plt.grid()
plt.xlabel("Days")
plt.ylabel("Rate of infection over days")
plt.legend()
plt.show()
