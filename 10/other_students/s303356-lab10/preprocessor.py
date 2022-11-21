import matplotlib.pyplot as plt
import pandas as pd
import math

# I considered 365 days in a year
N = 365

COLUMNS_PDF = ["dist", "days", "runs", "m", "prob", "ci_left", "ci_right"]
COLUMNS_EV = ["dist", "runs", "average", "ci_left", "ci_right"]

CONFIDENCE = 0.95

def column_keys(series):
    ''' Return an array '''
    return series.unique()

def analyze_curves(file_name="pdfs.txt", distribution='uniform'):
    '''
    :param file_name: File to load
    :return: null
    '''
    print(f"\n ** Pdfs comparison plot ({distribution})**")
    df = pd.read_csv(file_name, sep=",", header=None)
    df.rename(columns={0: COLUMNS_PDF[0], 1: COLUMNS_PDF[1], 2: COLUMNS_PDF[2], 3: COLUMNS_PDF[3], 4: COLUMNS_PDF[4], 5: COLUMNS_PDF[5], 6: COLUMNS_PDF[6]}, inplace=True)
    # For every configuration runs, N is fixed.
    for i in column_keys(df["runs"]):
        single_run = df[(df["runs"] == i) & (df["dist"] == distribution)]
        compare_curves(single_run["dist"].values[0], i, N, single_run["m"].values, single_run["prob"].values, single_run["ci_left"].values, single_run["ci_right"].values)
    return

def compare_curves(dist, runs, days, index, values, lower_values, upper_values):
    '''
    :param runs: Number of runs for each m
    :param days: Days considered
    :param index: Number of people considered
    :param values: Probability for each m
    :param lower_values: Left confidence interval
    :param upper_values: Right confidence interval
    :return: null
    '''
    plt.plot(index, values, 'g', label="Generated dist.")
    theoretical = []
    for i in index:
        theoretical.append(1 - (math.e**(- i**2 / (2*N))))
    plt.plot(index, theoretical, 'b--', label="Theoretical dist.", alpha=0.7)
    plt.plot(index, lower_values, 'r-.', alpha=0.4)
    plt.plot(index, upper_values, 'r-.', alpha=0.4)
    plt.fill_between(index, lower_values, upper_values, label=f"Confidence level {CONFIDENCE}", color='r', alpha=.1)
    plt.ylabel("Prob (birthday collision)")
    plt.xlabel("m (number of people)")
    plt.title(f" Number of experiment from each m: {runs}  days: {days} distribution: {dist}")
    plt.legend()
    plt.savefig(f'{dist}_{runs}.png')
    plt.show()
    return

def analyze_expected_value(file_name = "ev.txt"):
    '''
    :param file_name: File to load
    :return: null
    '''
    print(f"\n ** Average number of people to observe a conflict comparison**")
    df = pd.read_csv(file_name, sep=",", header=None)
    df.rename(columns={0: COLUMNS_EV[0], 1: COLUMNS_EV[1], 2: COLUMNS_EV[2], 3: COLUMNS_EV[3], 4: COLUMNS_EV[4]}, inplace=True)
    for i in df.values:
        print(f" * Runs: {int(i[1])}   Empirical average({i[0]}): {i[2]}   C.I.: [ {round(i[3],3)}, {round(i[4], 3)} ]")
    print(f" ** Theoretical average with N:{N} is {round(1.25*math.sqrt(N),3)} **")
    return

if __name__ == '__main__':
    analyze_curves(distribution="uniform")
    analyze_curves(distribution="realistic")
    analyze_expected_value()