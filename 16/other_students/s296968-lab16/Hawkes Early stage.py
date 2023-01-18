import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

T = 100  # time horizon
def simulate_epidemic(h):
    # Parameters
    T = 100 # simulation time horizon
    m = 2 # average number of secondary events caused by each primary event
    infected = 0
    deaths = 0
    events = []
    infected_over_time = [0] * T
    deaths_over_time = [0] * T
    # Generate initial events (seeds)
    # Assume 2% of individuals get infected and die
    population = 100000 # total population
    n_seeds = int(population * 0.02)  #number of people that are going to be infected. rounded to not have a fraction
    #The result is an array of n_seeds random numbers,
    #representing the time when the initial infected individuals were infected.
    seeds = np.random.uniform(0, T, size=n_seeds)
    #appends the seeds to the list events
    events.extend(seeds)
    # Simulate process over time horizon
    for t in range(T):
        lambda_t = m * h[t]
        for event_time in events:
            if event_time > t-10 and event_time <= t:
                lambda_t += 20 #we add sigma
        if lambda_t > 0:
            next_event = np.random.exponential(1/lambda_t)
            if t + next_event < T:
                events.append(t + next_event)
                infected += 1
                infected_over_time[t] = infected/population * 100 #to have a percentage
                deaths += np.random.binomial(infected, 0.02)
                deaths_over_time[t] = deaths/population * 100
    return infected_over_time, deaths_over_time

#Uniform h(t)
h1 = np.random.uniform(0, 20, T)

#Exponential h(t)
lambda_ = 1/10
h2 = lambda_ * np.exp(- lambda_ * np.arange(0, T, 1))

#Number of runs
n_runs = 1

#Initialize lists to store the results for uniform h(t)
all_infected_h1 = []
all_deaths_h1 = []
for i in range(n_runs):
    infected_over_time, deaths_over_time = simulate_epidemic(h1)
    all_infected_h1.append(infected_over_time)
    all_deaths_h1.append(deaths_over_time)

population = 1000000
#Plot the number of infected individuals for uniform h(t)
#labels = [f'Run {i+1}' for i in range(n_runs)]
plt.figure()
for i in range(n_runs):
    plt.plot(range(T), all_infected_h1[i], label='Infected')
for i in range(n_runs):
    plt.plot(range(T), all_deaths_h1[i], label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Percentage of Population')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.title('Number of infected/death individuals over time for uniform h(t)')
plt.legend()


#Initialize lists to store the results for exponential h(t)
all_infected_h2 = []
all_deaths_h2 = []

for i in range(n_runs):
    infected_over_time, deaths_over_time = simulate_epidemic(h2)
    all_infected_h2.append(infected_over_time)
    all_deaths_h2.append(deaths_over_time)

#Plot the number of infected individuals for exponential h(t)
plt.figure()
for i in range(n_runs):
    plt.plot(range(T), all_infected_h2[i], label='Infected')
for i in range(n_runs):
    plt.plot(range(T), all_deaths_h2[i], label='Infected')
plt.xlabel('Time (days)')
plt.ylabel('Percentage of Population')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.title('Number of infected/Deaths individuals over time for exponential h(t)')
plt.show()
