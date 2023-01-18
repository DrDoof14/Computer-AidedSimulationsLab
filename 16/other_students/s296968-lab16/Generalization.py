import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad
from scipy.optimize import minimize

# Parameters
sigma = 20 # Indicator function for t in [0, 10]
m = 2 # average number of secondary events caused by each primary event
T = 365 # simulation time horizon
lambda_ = 1/10 # decay rate for h(t)

def simulate_npi(n_runs):
    # Initialize lists to store results
    infected_list = []
    deaths_list = []
    cost_list = []
    rho_square_list = []
    for i in range(n_runs):
        # Initialize infected, death, cost and rho_square counters
        infected = 0
        deaths = 0
        cost = 0
        rho_square = 0
        infected_over_time = [0]*T
        deaths_over_time = [0] * T
        cost_over_time = [0]*T
        rho_square_over_time = [0] * T
        total_cost_over_time = [0] *T

        # Generate initial events (seeds)
        population = 100000 # total population
        n_seeds = int(population * 0.02)
        seeds = np.random.uniform(0, T, size=n_seeds)
        events = seeds.tolist()
        # Simulate process over time horizon
        for t in range(T):
            # Compute conditional intensity function
            lambda_t = m * lambda_ * np.exp(- lambda_ * t)
            # Start with rho(t)=1 for t <60
            rho = 1 if t < 60 else rho
            if t >= 60:
                if infected >=0 and infected < 30000:
                    rho = 0.3
                elif infected > 30000 and infected < 50000:
                    rho = 0.6
                elif infected >= 50000 and infected <80000:
                    rho = 0.8
                else:
                    rho = 1
            lambda_t = lambda_t * rho
            # Generate next event
            event_time = t + np.random.exponential(1/lambda_t)
            events.append(event_time)
            infected += 1
            infected_over_time[t] = infected/population * 100
            deaths += np.random.binomial(infected, 0.02)
            deaths_over_time[t]=deaths/population * 100
            cost += rho**2
            cost_over_time[t]=cost
            rho_square += rho**2 * (t+1) ## Multiplying by the number of days You are summing up all the rho_square
            # values at each day and the final value you get is the total cost, but in this model, since the rho is constant
            # over the entire year, so the cost will also be constant over the entire year as well.
            rho_square_over_time = rho_square

        infected_list.append(infected)
        deaths_list.append(deaths)
        cost_list.append(cost)
        rho_square_list.append(rho_square)
        average_death = sum(deaths_list) / n_runs

        if average_death > 20000:
            print("The average number of deaths exceed the threshold")
        return rho_square_over_time, infected_over_time, deaths_over_time, cost_over_time

# Number of runs
n_runs = 10
# Run the simulation
rho_square_over_time, infected_over_time, deaths_over_time, cost_over_time= simulate_npi(n_runs)

# Average cost of NPIs
average_cost = np.sum(rho_square_over_time, axis=0) / n_runs

# Integrate the rho_square over the time horizon
total_cost = [0] *T
for t in range(n_runs):
    total_cost[t] = np.sum(rho_square_over_time, axis=0)/n_runs

# Plot the number of infected and death individuals over time
plt.plot(range(T), infected_over_time, label="infected")
plt.plot(range(T), deaths_over_time, label="deaths")
plt.xlabel("Time (days)")
plt.ylabel('Percentage of Population')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.legend()
plt.show()

# Plot the total cost over time
plt.plot(range(T), cost_over_time)
plt.xlabel("Time (days)")
plt.ylabel("Cost")
plt.show()

print("Average cost: ", average_cost)




