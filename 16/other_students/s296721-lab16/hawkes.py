# %% [markdown]
# ''' 
# Simulate  an epidemic process at early stage, by using Hawkes processes with the following parameters:
# 
# \sigma(t)=20* \Ind(t \in [0,10])  days;
# 
#  either h(t) =  unif [0,20]  or h(t) = \lambda exp(- lambda t )  with \lambda=1/10 days;
# 
# m= 2.
# 
# Assume that 2% of individuals that gets infected,  die after a while, while the others  recover.
# 
# First, simulate the previous process on a time-horizon of  100 days (i.e. t\in [0, 100]),  using  the method that exploits  the inner branching structure of a Hawkes process.
# 
# Produce  some plots that show the evolution over time of infected/death individuals for the two choices of h(t) .
# 
# 
# 
# Then, generalise your analysis assuming that  starting from t=20 non pharmaceutical interventions  can be introduced  (with the effect of  reducing the stochastic intensity of the process of a factor rho(t), which may be dynamically adjusted on day by day basis).   
# 
# Assume that restrictions lead to  a per-day social/economical cost which is proportional  to \rho(t)^2 (i.e. the total cost is proportional to \int \rho^2(t) d t ).
# 
# Try to design  a strategy that attempts the minimisation the cost over a 1-year horizon, under the constraint that the average number of deaths should not exceed 20K over the whole year. 
# 
# In this case, you can assume h(t) either exponential or uniform according to your preference. 
# 
# Produce some plots that show the evolution over time of infected/death individuals and the obtained cost over 1 year  (i.e., t=[0,365]) .
# '''

# %% [markdown]
# # Simulate  an epidemic process at early stage

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
np.random.seed(66)

# Define the cost function
def cost_func(r):
  return r ** 2 # Cost is proportional to rho^2

# Define parameters of stochastic intensity
def sigma(t):
    if t >= 0  and t <= 10:
        return 20
    else:
        return 0
      
def h_uniform(t):
      return np.random.uniform(0,20)
    
def h_expo(t):
      return 0.1 * np.exp(-0.1 * t)

# Define Hawkes process

def simulate_hawkes_process(h_function, alpha, t_start, Ti):
    np.random.seed(66)
    
    T_list = []  # initialize empty set of event times
    s = 0   # initialize current time
    n = 0   # initialize number of events
    
    t_start = t_start # initialize start time of intervention
    target_death = 20000  # Average number of deaths over 1 year (20K)
    
    n_infected = 1 # initialize number of infected people
    last_children = 1 # initilaize number of children in each level of the tree
    
    # Define list for storing infected, death and the cost for the all process
    infected_list = []
    death_list = []
    cost_list = []
    cum_h = []
    
    rho = 1 # initialize reducing factor of stochastic intensity
    flag = 1 # initialize a flag to show when intervention ocurres
    
    
    while s < Ti:
        # compute current intensity
        lambda_s = sigma(s) + alpha*sum(h_function(s - t) for t in T_list)
        
        # start intervention if we pass t_start
        if (s >= t_start) and flag:
            
            '''
            Note that this code uses the scipy.optimize.minimize function to optimize the value of rho using the SLSQP algorithm, 
            which is a method for quadratic optimizing with bounds and constraints. 
            The cost function is used as the objective function, 
            and a constraint is added to ensure that the average number of deaths does not exceed the death threshold. 
            the constraint is the diffrence between the rate of current death number and maximum allowed death number
            
            '''
            # Find minimum based on cost function and the constraint and create rho which proportional to the minimum
            rho = 1e5 * 0.25 * minimize_scalar(lambda rho: cost_func(rho) + (Ti-s)/Ti * (death_list[-1]/s - target_death/Ti)**2, bounds=(0, 1), method='bounded').x
            flag = 0
        
        
        # generate candidate event time which should be integer and >=1 based on assigned distribution
        if h_function == h_uniform:
            w = int(np.random.uniform(1, 11))
        else:
            w = int(np.random.exponential(alpha)) + 1

        # generate next event time (point process)
        s += w
        
        # accept candidate event time with probability lambda_prob / lambda_s
        D = np.random.uniform()
        lambda_prob = rho * (sigma(s) + alpha*sum(h_function(s - t) for t in T_list))
        
        # perform thin process
        if D <=(lambda_prob / lambda_s):
            
            # generate children of each member in the current level for next level of tree based on poisson dist. and average number of infected (alpha) 
            temp = 0 
            for i in range(last_children):
                temp += np.random.poisson(alpha)
            
            
            last_children = temp # store children number of current level for next computation
            
            n_infected += last_children  # update number of events
            T_list.append(s)  # add event time to list
            
            n_death = int(0.02 * n_infected) # compute death number of each level
            
            # Update lists
            infected_list.append(n_infected - n_death)
            death_list.append(n_death)
            cum_h.append(lambda_prob)
            
            # compute cost whenever intervention starts and add it to cost list
            if flag:
                cost_list.append(0)
            else:
                # since we have district time we use sum instead of integral
                cost_list.append(cost_list[-1] + cost_func(rho))
                
        
    # return set of event times
    if T_list[-1] <= Ti:
        print('first')
        return T_list, infected_list, death_list, cost_list, cum_h
    else:
        print('second')
        return T_list[:-1], infected_list[:-1], death_list[:-1], cost_list[:-1], cum_h[:-1]

def plotting(event_times, infected_list, death_list, cost_list, T, t_start, h_function):
    
    if t_start >= T:
       if h_function == h_expo:
           temp = 'Exponential case without intervention'
       else:
           temp = 'Uniform case without intervention'
    else:
        if h_function == h_expo:
            temp = 'Exponential case with intervention'
        else:
            temp = 'Uniform case with intervention'
           
    # Plot the evolution over time of infected/death individuals
    plt.figure()
    plt.plot(event_times, infected_list, label='Infected')
    plt.scatter(event_times, [0] * len(event_times))
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title(temp)
    plt.legend()
    plt.savefig(f'{temp} Infected.png')
    plt.show()
    

    plt.figure()
    plt.plot(event_times, death_list, label='Death')
    plt.scatter(event_times, [0] * len(event_times))
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title(temp)
    plt.legend()
    plt.savefig(f'{temp} Death.png')
    plt.show()
    

    plt.figure()
    plt.plot(event_times, cost_list, label='Cost')
    plt.scatter(event_times, [0] * len(event_times))
    plt.xlabel('Time (days)')
    plt.ylabel('Cost Function')
    plt.title(temp)
    plt.legend()
    plt.savefig(f'{temp} Cost.png')
    plt.show()
    

# %% [markdown]
# # An illustrative example of the left-continuous conditional intensity λ(t) when h(t) has uniform distribution

# %%
T = 50 # define t_horizontal 
#Define simulation parameters
h_function = h_uniform
alpha = 2
t_start = T + 1
Ti = T

event_times, infected_list, death_list, cost_list, cum_h = simulate_hawkes_process(h_function, alpha, t_start, Ti)

plt.figure(figsize=(15,2))
plt.step(event_times, np.cumsum(np.ones_like(cum_h)))
plt.ylabel("$\lambda^*(t)$")
plt.xlabel('$t$')
plt.title('Conditional Intensity _ Uniform Case')
_ = plt.plot(event_times, np.zeros_like(cum_h), 'k.')
plt.savefig('lambda_uniform.png')
plt.show()


# %% [markdown]
# # An illustrative example of the left-continuous conditional intensity λ(t) when h(t) has exponential iform distribution

# %%
T = 50 # define t_horizontal
#Define simulation parameters
h_function = h_expo
alpha = 2
t_start = T + 1
Ti = T
 
event_times, infected_list, death_list, cost_list, cum_h = simulate_hawkes_process(h_function, alpha, t_start, Ti)


smp = np.asarray(event_times)
range_list = np.arange(0, T, .001)
lda_ar = [0.1 + np.sum(h_expo(x - smp[smp < x])) for x in range_list]

plt.figure(figsize=(15,2))
plt.ylabel("$\lambda^*(t)$")
plt.xlabel("$t$")
plt.title('Conditional Intensity _ Exponential Case')
plt.yticks(np.arange(0, 5, 0.1))
_ = plt.plot(range_list, lda_ar, 'b-')
plt.scatter(event_times, [0] * len(event_times))
plt.savefig('lambda_expo.png')
plt.show()


# %% [markdown]
# # Evolution over time of infected/death individuals for the exponential h(t) without intervention

# %%
T = 100 # define t_horizantal
#Define simulation parameters
h_function = h_expo
alpha = 2
t_start = T + 1
Ti = T

event_times, infected_list, death_list, cost_list, cum_h = simulate_hawkes_process(h_function, alpha, t_start, Ti)
plotting(event_times, infected_list, death_list, cost_list, T, t_start, h_function)



# %% [markdown]
# # Evolution over time of infected/death individuals for the uniform h(t) without intervention

# %%
T = 100 # define t_horizantal
#Define simulation parameters
h_function = h_uniform
alpha = 2
t_start = T + 1
Ti = T

event_times, infected_list, death_list, cost_list, cum_h = simulate_hawkes_process(h_function, alpha, t_start, Ti)
plotting(event_times, infected_list, death_list, cost_list, T, t_start, h_function)

# %% [markdown]
# # Evolution over time of infected/death individuals for the exponential h(t) with intervention and maximum death of 20K

# %%
T = 365 # define t_horizantal
#Define simulation parameters
h_function = h_uniform
alpha = 2
t_start = 20
Ti = T

event_times, infected_list, death_list, cost_list, cum_h = simulate_hawkes_process(h_function, alpha, t_start, Ti)
plotting(event_times, infected_list, death_list, cost_list, T, t_start, h_function)


