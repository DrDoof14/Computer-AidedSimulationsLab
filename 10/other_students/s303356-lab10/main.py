import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import norm
import math

'''Random seeds considered for consistency of results is 70, 71, 72'''
random.seed(70)

# I considered 365 days in a year
N = 365

# Max number of students that I consider is 80. The limit is up to 365 (MAX_M < N)
MAX_M = 80

# Min number of student that I consider is 100
MIN_M = 2

# Runs for avery m selected.
RUNS_PROB = [100, 1000]

# Generate average number of people to observe a conflict. Average in function of number of iteration.
RUNS_E = 10, 20, 40

# Level of confidence - alpha= 0.1
CONFIDENCE = 0.95

class Distribution:
    def __init__(self):
        '''
        Init the classes loading the csv representing a samples of the real distribution of birthday, extrapolate and
        save it on a list
        '''
        df = pd.read_csv("birthdays.csv")
        # Dates = (1969-01-01 - 1988-12-31) - source = https://calmcode.io/birthday-problem/dataset.html
        plot_df = (df.assign(date=lambda d: pd.to_datetime(d['date']))
                   .assign(day_of_year=lambda d: d['date'].dt.dayofyear)
                   .groupby('day_of_year')
                   .agg(n_births=('births', 'sum'))
                   .assign(p=lambda d: d['n_births'] / d['n_births'].sum()))
        self.dist = plot_df["p"].values
        # Need to adjust the distribution - exclude the last overlapping day and checked the sum to 1.
        self.dist = self.dist[:len(self.dist)-1]
        self.dist[-1] = 1-sum(self.dist[:-1])
        self.width = 0.8

    def show_dist(self):
        plt.bar(range(len(self.dist)), self.dist, width=self.width, facecolor='g', alpha=0.75)
        plt.xlabel('Day')
        plt.ylabel('probability of birthday')
        plt.title('Probability of the birthday from U.S. (1969 - 1988)')
        plt.grid(True)
        plt.show()
        return

    def get_dist(self):
        return self.dist

dist = Distribution()

def check_conflict(list):
    '''
    :param list: list of numbers to check
    :return: true if a conflict is found, false otherwise
    '''
    days = set()
    last_len = 0
    for i in list:
        days.add(i)
        if last_len != len(days):
            last_len = len(days)
        else:
            return True
    return False

def create_dist(save_on_file=False, distribution = "uniform"):
    '''
    :param save_on_file: True if you want to save the results in the file
    :return: list containing the tuples : (runs, m, prob, ci_left, ci_right) for every combinations of runs and m
    '''
    print(f"\n ** Creating probability distr. ({distribution}) with parameters > MIN_M: {MIN_M}  MAX_M: {MAX_M}  RUNS: {RUNS_PROB}  DAYS: {N} **")
    for z in RUNS_PROB:
        print(f"  * Computing each M {z} times *")
        prob_list = []
        for i in range(MIN_M, MAX_M + 1):
            count = 0
            for j in range(z):
                if distribution == "uniform":
                    day = np.random.uniform(1, N + 1, i)
                else:
                    day = np.random.choice(a=range(1, N+1), size=i, p=dist.get_dist())
                day = [int(i) for i in day]
                if (check_conflict(day)):
                    count += 1
            p = count/z
            s = math.sqrt((p*(1-p)) / z)
            interval = [p-(norm.ppf(CONFIDENCE+(1-CONFIDENCE)/2)) * s, p+(norm.ppf(CONFIDENCE+(1-CONFIDENCE)/2)) * s]
            prob_list.append((z, i, p, interval[0], interval[1]))
        if save_on_file == True:
            '''Saving format : (dist, DAYS CONSIDERED, RUNS, M, PROB, CI_LEFT, CI_RIGHT)'''
            save_dist(distribution, N, prob_list)
    return prob_list

def save_dist(distribution, days, pdf_list):
    '''
    :param days: Days considered
    :param pdf_list: list containing (runs, m, prob, ci_left, ci_right) tuples
    :return: null
    '''
    try:
        with open("pdfs.txt", "a") as output:
            for row in pdf_list:
                output.write(distribution + ',' + str(days) + ',' + str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' + str(row[4])+'\n')
    except:
        print(f"Can't save the file, something went wrong")
    return

def create_expected_value(save_on_file=False, distribution = "uniform"):
    ''' Run the experiment to compute the average number of people to observe a conflict.
    :param save_on_file: True if you want to save the results in the correspondent file
    :return: null
    '''
    print(f"\n ** Computing average number of people to observe a conflict ({distribution} dist. )**")
    results = []
    for z in RUNS_E:
        print(f"  * Number of runs {z}")
        conflict = []
        for i in range(z):
            days = []
            for j in range(MIN_M, MAX_M + 1):
                if distribution == "uniform":
                    day = np.random.uniform(1, N + 1)
                else:
                    day = np.random.choice(a=range(1, N+1), p=dist.get_dist())
                days.append(int(day))
                days = [int(a) for a in days]
                if (check_conflict(days)):
                    conflict.append(j)
                    break
        p = sum(conflict)/len(conflict)
        std = math.sqrt(get_variance(conflict))
        interval = t.interval(1-CONFIDENCE, z - 1, p, std)
        results.append((distribution, z, p, interval[0], interval[1]))
    if save_on_file == True:
        '''Saving format: (RUNS, Expected value, left_interval, right_interval)'''
        save_expected_value(results)
    return

def get_variance(lista):
    '''
    :param lista: list of numbers
    :return: variance of the list
    '''
    avg = sum(lista) / len(lista)
    acc = 0
    for i in lista:
        acc += (i-avg)**2
    return acc/(len(lista) - 1)

def save_expected_value(expected_value_list):
    '''
    :param expected_value_list: list of tuples (RUNS, Expected value, left_interval, right_interval), RUNS is the time
    that the esperiment is run
    :return: null
    '''
    try:
        with open("ev.txt", "a") as output:
            for row in expected_value_list:
                output.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + '\n')
    except:
        print(f"Can't save the file, something went wrong")
    return

if __name__ == '__main__':
    # Reset files
    open("pdfs.txt", "w")
    open("ev.txt", "w")
    # Metrics computed using both distribution
    create_dist(save_on_file=True)
    create_dist(save_on_file=True, distribution="realistic")
    create_expected_value(save_on_file=True)
    create_expected_value(save_on_file=True, distribution="realistic")