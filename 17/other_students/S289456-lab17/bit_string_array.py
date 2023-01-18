import random
import Functions
from matplotlib import pyplot as plt


def bit_string_array():
    random.seed(32)
    sentence_set = Functions.sentence_from_text(sentence_size=6)
    m = len(sentence_set)
    b_list = [*range(19, 24, 1)]
    prob_FP_sim = []
    memory_simulation = []
    # PART 01
    # calculating probabilities of FP and memory in Simulation
    for b in b_list:
        BSA = Functions.BitStringArray(size=pow(2, b))
        for sentence in sentence_set:
            BSA.add(sentence)
        prob_FP = BSA.count_ones() / pow(2, b)
        prob_FP_sim.append(prob_FP)
        memory_simulation.append(0.000125 * m / prob_FP)
    plt.plot(b_list, prob_FP_sim, label='Simulation')
    plt.scatter(b_list, prob_FP_sim)
    plt.xticks(ticks=b_list)

    # Part 02
    # calculating probabilities of FP in Theory
    prob_theory = []
    for b in b_list:
        prob_theory.append(m / pow(2, b))

    plt.plot(b_list, prob_theory, label='Theory')
    plt.scatter(b_list, prob_theory)
    plt.xlabel('b')
    plt.ylabel('False Positive Probability')
    plt.title('Comparison of Theory & Simulation (Bit string array)')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(b_list, memory_simulation)
    plt.scatter(b_list, memory_simulation)
    plt.xticks(ticks=b_list)
    plt.xlabel('b')
    plt.ylabel('Size(KB)')
    plt.title('Memory occupation in Simulation (Bit string array)')
    plt.grid()
    plt.show()
