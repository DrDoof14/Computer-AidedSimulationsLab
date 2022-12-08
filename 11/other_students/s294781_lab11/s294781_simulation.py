import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from numpy import random
from scipy.stats import norm, t
# Set the seed for reproducibility
random.seed(10)

# Number of servers
SERVERS = 2
# Maximum size of the clients queue
QUEUE_SIZE = 1000
# Threshold for the absolute difference of consecutive cumulative means in the
# transient analysis
TRANSIENT_TOLERANCE = 1e-3
# Threshold for number of consecutive samples with stable cumulative means in
# the transient analysis
TRANSIENT_LENGTH_STEADY = 50
# Duration of the simulation after the transient end measured as number of
# delays collected
SIM_LENGTH = 2000


# Returns the mean and the confidence interval for it of elements
def confidence_interval(elements, confidence):
  n = len(elements) # Number of elements
  mean = np.mean(elements) # Mean of the elements
  # Standard deviation estimation of the elements unbiased (divided by n - 1)
  std = np.std(elements, ddof=1)
  # If there are less than 30 samples use t Student
  if n < 30:
    # Return mean and confidence interval computed using the t Student
    return mean, t.interval(confidence, n-1, mean, std / sqrt(n))
  else:
    # Return mean and confidence interval computed using the normal distribution
    return mean, norm.interval(confidence, mean, std / sqrt(n))

# Create new plot figure
def new_plot(xlabel, ylabel):
  fig, ax = plt.subplots()
  ax.set_xlabel(xlabel) # Set label x axis
  ax.set_ylabel(ylabel) # Set label y axis
  return ax

# Class containing client information
class Client:
  # Initialize the client with its arrival time and the service time distribution
  def __init__(self, arrival_time, service_time):
    self.arrival_time = arrival_time
    # Sample a service time from the given distribution
    self.service_time = service_time()
  # Start serving the client, returns the end time
  def serve(self, time):
    self.start_serve = time
    return time + self.service_time
  # Returns the expected end time
  def get_end_time(self):
    return self.start_serve + self.service_time
  # Manage client service finished event, compute and return the delay
  def end_service(self, served_time):
    self.delay = served_time - self.arrival_time
    return self.delay
  # Manage the premature stopping of the service for preemption
  def preemption(self, time):
    # Decrease the service time of the time it has being served
    self.service_time = self.service_time - (time - self.start_serve)

# Class managing the queue
class Queue:
  def __init__(self, service_time_high_priority, service_time_low_priority, lambda_arrival):
    self.queue_high_priority = list() # Queue of clients with high priority
    self.queue_low_priority = list() # Queue of clients with low priority
    self.fes = list() # Future Event Set
    self.users_served_high_priority = 0 # Total high priority users currently being served
    self.users_served_low_priority = 0 # Total low priority users currently being served
    # Insert arrival of first high priority client
    self.fes.append((0, "arrival_high_priority"))
    # self.queue_high_priority.append(Client(0, service_time_high_priority))
    # Insert arrival of first low priority client
    self.fes.append((0, "arrival_low_priority"))
    # self.queue_low_priority.append(Client(0, service_time_low_priority))
    self.users = 0 # Total users in the queue
    # Distribution used for the service time of high priority clients
    self.service_time_high_priority = service_time_high_priority
    # Distribution used for the service time of low priority clients
    self.service_time_low_priority = service_time_low_priority
    self.lambda_arrival = lambda_arrival # Rate of arrival in the queue
  
  # Return next arrival time
  def next_arrival(self, time):
    # The arrival distribution is a poisson with rate lambda_arrival
    return time + random.exponential(1 / self.lambda_arrival)
  
  # Return the next event in the queue
  def next_event(self):
    self.fes.sort() # Sort the events in increasing order of time
    return self.fes.pop(0)
  
  # Serve a client if there are servers available or replace low priority client
  # with high priority one
  def serve_next(self, time):
    # Compute the total number of users currently being served
    total_users_served = self.users_served_low_priority + self.users_served_high_priority
    # If there is an empty server and there is a client waiting, process the client
    if total_users_served < SERVERS and self.users > total_users_served:
      # If there is an high priority client waiting, serve it
      if len(self.queue_high_priority) > self.users_served_high_priority:
        self.fes.append((
            self.queue_high_priority[self.users_served_high_priority].serve(time), 
            "departure_high_priority"))
        self.users_served_high_priority += 1
      # If there is an low priority client waiting, serve it
      elif len(self.queue_low_priority) > self.users_served_low_priority: 
        self.fes.append((
            self.queue_low_priority[self.users_served_low_priority].serve(time), 
            "departure_low_priority"))
        self.users_served_low_priority += 1
    # If there is an high priority client waiting and a low priority client being
    # served, replace it
    if len(self.queue_high_priority) > self.users_served_high_priority and self.users_served_low_priority > 0:
      # Get expected finish time to remove departure from FES
      end_time = self.queue_low_priority[self.users_served_low_priority - 1].get_end_time()
      # Remove previously scheduled departure from FES
      self.fes.remove((end_time, "departure_low_priority"))
      # Preempts the last low priority users served
      self.queue_low_priority[self.users_served_low_priority - 1].preemption(time)
      self.users_served_low_priority -= 1
      # Serve the high priority client
      self.fes.append((
            self.queue_high_priority[self.users_served_high_priority].serve(time), 
            "departure_high_priority"))
      self.users_served_high_priority += 1
  
  # Manage the arrival of a low priority client
  def arrival_low_priority(self, time):
    # Compute the arrival time
    next_arrival_time = self.next_arrival(time)
    # Add the arrival event and the client to the lists
    self.fes.append((next_arrival_time, "arrival_low_priority"))
    # If there is still space in the queue, add client
    if self.users < QUEUE_SIZE:
      self.queue_low_priority.append(Client(time, self.service_time_low_priority))
      self.users += 1 # Increase the number of clients in the queue
    # Serve a client if possible
    self.serve_next(time)
  
  # Manage the arrival of an high priority client
  def arrival_high_priority(self, time):
    # Compute the arrival time
    next_arrival_time = self.next_arrival(time)
    # Add the arrival event and the client to the lists
    self.fes.append((next_arrival_time, "arrival_high_priority"))
    # If there is still space in the queue, add client
    if self.users < QUEUE_SIZE:
      self.queue_high_priority.append(Client(time, self.service_time_high_priority))
      self.users += 1 # Increase the number of clients in the queue
    # If there is no space, try replace a low priority client with the current
    # high priority client
    elif len(self.queue_low_priority) > 0:
      # If all the low priority users are being served, preempts the last one before deleting it
      if len(self.queue_low_priority) == self.users_served_low_priority:
        # Get expected finish time to remove departure from FES
        end_time = self.queue_low_priority[self.users_served_low_priority - 1].get_end_time()
        # Remove previously scheduled departure from FES
        self.fes.remove((end_time, "departure_low_priority"))
        self.users_served_low_priority -= 1 # Decrease the number of low priority users served
      # Delete last low priority client
      self.queue_low_priority.pop(-1)
      # Add high priority client
      self.queue_high_priority.append(Client(next_arrival_time, self.service_time_high_priority))
    # Serve a client if possible
    self.serve_next(time)

  # Manage the departure of a low priority client, returns the delay of the client
  def departure_low_priority(self, time):
    # Remove the client with the least departure time
    end_times_served = [client.get_end_time() for client in self.queue_low_priority[:self.users_served_low_priority]]
    client = self.queue_low_priority.pop(np.argmin(end_times_served)) # Remove the client from the queue
    delay_time = client.end_service(time) # Notify client of end service and get the delay
    self.users -= 1 # Decrease the number of clients in the queue
    self.users_served_low_priority -= 1 # Decrease the number of served users with low priority
    self.serve_next(time) # Serve a client if possible
    return delay_time
  
  # Manage the departure of a high priority client, returns the delay of the client
  def departure_high_priority(self, time):
    # Remove the client with the least departure time
    end_times_served = [client.get_end_time() for client in self.queue_high_priority[:self.users_served_high_priority]]
    client = self.queue_high_priority.pop(np.argmin(end_times_served)) # Remove the client from the queue
    delay_time = client.end_service(time) # Notify client of end service and get the delay
    self.users -= 1 # Decrease the number of clients in the queue
    self.users_served_high_priority -= 1 # Decrease the number of served users with high priority
    self.serve_next(time) # Serve a client if possible
    return delay_time

  # Return the number of clients in the queue
  def get_size(self):
    return self.users

# Class managinig the simulation of the queue
class Simulation:
  def __init__(self, service_time_high_priority, service_time_low_priority, lambda_arrival):
    print(lambda_arrival)
    self.lambda_arrival = lambda_arrival # Rate of arrrival in the queue
    # Distribution used for the service time of high priority clients
    self.service_time_high_priority = service_time_high_priority
    # Distribution used for the service time of low priority clients
    self.service_time_low_priority = service_time_low_priority
  # Save the delay of a client
  def store_delay(self, delay):
    self.total_delays.append(delay) # Append the delay
    # Compute the new cumulative average
    n = len(self.cumulative_avg_delays)
    if n > 0:
      # Update the previous average with the current delay
      new_avg = (n*self.cumulative_avg_delays[-1] + delay) / (n + 1)
    else:
      new_avg = delay
    self.cumulative_avg_delays.append(new_avg) # Append the new cumulative average
  # Remove all the samples beloning to the transient
  def remove_transient(self):
    self.total_delays = list() # Initialize the list storing the delay of each sample
    self.cumulative_avg_delays = list() # Initialize the list storing the cumulative delay averages
    transient = True # Flag indicating if we are in the transient
    steady_avg = 0 # Counters for the number of samples with stable cumulative delay averages
    while transient: # Loop until transient is not finished
      (time, event) = self.queue.next_event() # Get next queue event
      if event == "arrival_high_priority": # Arrival of an high priority client
        self.queue.arrival_high_priority(time) # Notify the queue
      elif event == "arrival_low_priority": # Arrival of a low priority client
        self.queue.arrival_low_priority(time) # Notify the queue
      # Departure of a client, for transient removal the delays for both priorities
      # are treated the same
      elif event == "departure_high_priority" or event == "departure_low_priority":
        # Call the rigth departure depending on the priority
        if event == "departure_high_priority":
          delay_time = self.queue.departure_high_priority(time) # Depart from the queue and get delay
        elif event == "departure_low_priority":
          delay_time = self.queue.departure_low_priority(time) # Depart from the queue and get delay
        self.store_delay(delay_time) # Save the delay and compute cumulative average
        if len(self.cumulative_avg_delays) > 1: # If there are at least two delays
          # Compute difference of last two cumulative delays
          difference = abs(self.cumulative_avg_delays[-2] - self.cumulative_avg_delays[-1])
          # If the difference is less than the threshold, we consider it stable
          if difference < TRANSIENT_TOLERANCE:
            steady_avg += 1 # Increase the number of stable samples
            # If the number of stable samples if above the threshold, then the transient is finished
            if steady_avg >= TRANSIENT_LENGTH_STEADY:
              transient = False # Set transient as finished
              transient_end_time = time
          else: # If the threshold is not respected, reset the number of stable samples
            steady_avg = 0
    # Store the number of the last sample in the transient
    self.transient_end = len(self.cumulative_avg_delays)
    # Return the time of the last event in the transient
    return transient_end_time
  # Perform the simulation and return number of batches, average delay and its confidence interval
  def simulate(self):
    time = 0 # Initialize the simulation time
    # Instantiate the queue
    self.queue = Queue(self.service_time_high_priority, self.service_time_low_priority, self.lambda_arrival)
    self.delays_all = list() # Initialize the list storing all the delays
    self.delays_high_priority = list() # Initialize the list storing the delays of high priority users
    self.delays_low_priority = list() # Initialize the list storing the delays of low priority users
    self.remove_transient() # Remove transiens
    while len(self.delays_all) < SIM_LENGTH: # Loop until SIM_LENGTH delays are collected
      (time, event) = self.queue.next_event() # Get next queue event
      if event == "arrival_high_priority": # Arrival of an high priority client
        self.queue.arrival_high_priority(time) # Notify the queue
      elif event == "arrival_low_priority": # Arrival of a low priority client
        self.queue.arrival_low_priority(time) # Notify the queue
      elif event == "departure_high_priority": # Departure of an high priority client
        delay_time = self.queue.departure_high_priority(time) # Depart from the queue and get delay
        self.delays_all.append(delay_time) # Append the delay to the general list
        self.delays_high_priority.append(delay_time) # Append the delay to the high priority list
        self.store_delay(delay_time) # Save the delay and compute cumulative average for plotting transient
      elif event == "departure_low_priority": # Departure of a low priority client
        delay_time = self.queue.departure_low_priority(time) # Depart from the queue and get delay
        self.delays_all.append(delay_time) # Append the delay to the general list
        self.delays_low_priority.append(delay_time) # Append the delay to the low priority list
        self.store_delay(delay_time) # Save the delay and compute cumulative average for plotting transient
    return confidence_interval(self.delays_all, 0.95), confidence_interval(self.delays_high_priority, 0.95), confidence_interval(self.delays_low_priority, 0.95)
  # Plot the delays and cumulative averages of each sample with a line showing the transient end
  def plot_transient(self):
    ax = new_plot("Sample", "Delay") # Create new plot
    # Plot the delay for each sample
    ax.plot(self.total_delays, alpha=0.5, label="Delay")
    # Plot the cumulative delay for each sample
    ax.plot(self.cumulative_avg_delays, linewidth=2, label="Cumulative mean delay")
    # Plot the trasient end
    ax.vlines(self.transient_end, np.min(self.total_delays), np.max(self.total_delays), label="Transient end")
    ax.legend() # Show legend

# Initialize data structures for saving the results
results = pd.DataFrame(
    columns=["ServiceDistribution", "ExpectationHighProprity",
             "ExpectationLowProprity", "LambdaArrival", "DelayMeanAll",
             "ConfidenceInterval95LeftAll", "ConfidenceInterval95RightAll",
             "DelayMeanHighPriority", "ConfidenceInterval95LeftHighPriority",
             "ConfidenceInterval95RightHighPriority", "DelayMeanLowPriority",
             "ConfidenceInterval95LeftLowPriority", "ConfidenceInterval95RightLowPriority"])

# Different rates of arrival to simulate
lambdas_arrivals = [0.2, 0.4, 0.8, 1.4, 2.0, 2.4, 2.8]
# Simulate deterministic service time
print("Deterministic")
for lambda_arrivals in lambdas_arrivals:
  # Instantiate simulation class with deterministic service time and same
  # distribution for high and low priority with mean 1
  simulation_deterministic = Simulation(lambda: 1, lambda: 1, lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_deterministic.simulate()
  # Store the results
  results.loc[len(results)] = [
      "Deterministic", 1, 1, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_deterministic.plot_transient()
  # Instantiate simulation class with deterministic service time and the
  # distribution for high and low priority with mean 1/2 and 3/2
  simulation_deterministic = Simulation(lambda: 1/2, lambda: 3/2, lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_deterministic.simulate()
  # Store the results
  results.loc[len(results)] = [
      "Deterministic", 1/2, 3/2, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_deterministic.plot_transient()

# Simulate exponential service time
print("Exponential")
for lambda_arrivals in lambdas_arrivals:
  # Instantiate simulation class with exponential service time and same
  # distribution for high and low priority with mean 1
  simulation_exponential = Simulation(
      lambda: random.exponential(1), lambda: random.exponential(1), lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_exponential.simulate()
  # Store the results
  results.loc[len(results)] = [
      "Exponential", 1, 1, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_exponential.plot_transient()
  # Instantiate simulation class with exponential service time and the
  # distribution for high and low priority with mean 1/2 and 3/2
  simulation_exponential = Simulation(
      lambda: random.exponential(1/2), lambda: random.exponential(3/2), lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_exponential.simulate()
  # Store the results
  results.loc[len(results)] = [
      "Exponential", 1/2, 3/2, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_exponential.plot_transient()

hyper_exponential_parameters = {
    1: [(sqrt(2)-1)/sqrt(2), (99+sqrt(2))/sqrt(2)],
    1/2: [(2-sqrt(2))/4, (2+99*sqrt(2))/4],
    3/2: [(6-3*sqrt(2))/4, (6+297*sqrt(2))/4]
}
# Simulate an hyper-exponential with mean 1 and std 10 with 2 exponentials
def hyper_exponential(mean):
  # The means and weights are calculated such that the mean is 1 and std is 10
  exp_means = hyper_exponential_parameters[mean]
  weights = [0.99, 0.01]
  exp_choice = random.choice([0, 1], p=weights)
  return random.exponential(exp_means[exp_choice])

# Simulate hyper-exponential service time
print("Hyper-exponential")
for lambda_arrivals in lambdas_arrivals:
  # Instantiate simulation class with hyperexponential service time and same
  # distribution for high and low priority with mean 1
  simulation_hyperexponential = Simulation(
      lambda: hyper_exponential(1), lambda: hyper_exponential(1), lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_hyperexponential.simulate()
  # Store the results
  results.loc[len(results)] = [
      "HyperExponential", 1, 1, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_hyperexponential.plot_transient()
  # Instantiate simulation class with hyperexponential service time and the
  # distribution for high and low priority with mean 1/2 and 3/2
  simulation_hyperexponential = Simulation(
      lambda: hyper_exponential(1/2), lambda: hyper_exponential(3/2), lambda_arrivals)
  # Simulate
  delay_mean_all, delay_mean_hp, delay_mean_lp = simulation_hyperexponential.simulate()
  # Store the results
  results.loc[len(results)] = [
      "HyperExponential", 1/2, 3/2, lambda_arrivals, delay_mean_all[0], delay_mean_all[1][0],
      delay_mean_all[1][1], delay_mean_hp[0], delay_mean_hp[1][0], delay_mean_hp[1][1],
      delay_mean_lp[0], delay_mean_lp[1][0], delay_mean_lp[1][1]]
  # Plot the transient removal
  simulation_hyperexponential.plot_transient()

# PLOTTING
# Plot the mean delay for the distribution in function of the rate of arrival
def plot_delay(results, distribution, expectation): #, ax, color, marker):
  ax = new_plot("Rate of arrival", "Delay") # Create new plot
  ax.set_title(f"Service distribution: {distribution} High priority expectation: {expectation}")
  # Take the results of the selected distribution
  results_distribution = results[results["ServiceDistribution"] == distribution]
  # Take the results of the selected distribution with the expected value
  results_distribution_mean = results_distribution[results_distribution["ExpectationHighProprity"] == expectation]
  # Plot the curve of the average delay for all customers
  results_distribution_mean.plot(x="LambdaArrival", y="DelayMeanAll", ax=ax, label="All", marker="o") 
  # Plot the confidence interval
  ax.fill_between(
      x=results_distribution_mean["LambdaArrival"], 
      y1=results_distribution_mean["ConfidenceInterval95LeftAll"], 
      y2=results_distribution_mean["ConfidenceInterval95RightAll"],
      alpha=0.4)
  # Plot the curve of the average delay for high priority customers
  results_distribution_mean.plot(x="LambdaArrival", y="DelayMeanHighPriority", ax=ax, label="High priority", marker="o") 
  # Plot the confidence interval
  ax.fill_between(
      x=results_distribution_mean["LambdaArrival"], 
      y1=results_distribution_mean["ConfidenceInterval95LeftHighPriority"], 
      y2=results_distribution_mean["ConfidenceInterval95RightHighPriority"],
      alpha=0.4)
  # Plot the curve of the average delay for low priority customers
  results_distribution_mean.plot(x="LambdaArrival", y="DelayMeanLowPriority", ax=ax, label="Low priority", marker="o") 
  # Plot the confidence interval
  ax.fill_between(
      x=results_distribution_mean["LambdaArrival"], 
      y1=results_distribution_mean["ConfidenceInterval95LeftLowPriority"], 
      y2=results_distribution_mean["ConfidenceInterval95RightLowPriority"],
      alpha=0.4)
  ax.legend()

# Plot the delays of deterministic service time with the expectation of high
# priority customers equal 1
plot_delay(results, "Deterministic", 1)
# Plot the delays of deterministic service time with the expectation of high
# priority customers equal 1/2
plot_delay(results, "Deterministic", 1/2)
# Plot the delays of exponential service time with the expectation of high
# priority customers equal 1
plot_delay(results, "Exponential", 1)
# Plot the delays of exponential service time with the expectation of high
# priority customers equal 1/2
plot_delay(results, "Exponential", 1/2)
# Plot the delays of hyperexponential service time with the expectation of high
# priority customers equal 1
plot_delay(results, "HyperExponential", 1)
# Plot the delays of hyperexponential service time with the expectation of high
# priority customers equal 1/2
plot_delay(results, "HyperExponential", 1/2)

plt.show() # Show plots