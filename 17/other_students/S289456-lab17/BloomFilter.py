import math
import random
import numpy as np
import matplotlib.pyplot as plt
import Functions


def bloomfilter():
    # Seed the random number generator for reproducibility
    random.seed(32)
    np.random.seed(100)

    # Create a set of sentences
    sentence_set = Functions.sentence_from_text(sentence_size=6)
    m = len(sentence_set)
    # Part 01 DONE

    # Create a range of values for k
    k_range = [*range(1, 70)]
    # Create a range of values for b
    b_list = [*range(19, 24, 1)]

    # Create a figure for the plot
    plt.figure(figsize=(12, 8))

    # Initialize an empty list to store the optimal k values
    optimal_k = []
    # Iterate over the list of b values
    for b in b_list:
        prob_FP = []
        # Iterate over the list of k values
        for k in k_range:
            # Calculate the size of the Bloom filter
            n = pow(2, b)
            # Create a Bloom filter object
            BF = Functions.BloomFilter(size=n, hash_count=k)
            # Add each sentence to the Bloom filter
            for sentence in sentence_set:
                BF.add(sentence)
            # Count the number of ones in the Bloom filter
            ones = BF.count_ones()
            # Calculate the probability of false positives
            prob_FP.append(pow(ones / n, k))
        # Append the optimal k value and the corresponding false positive probability to the list
        optimal_k.append([prob_FP.index(min(prob_FP)) + 1, min(prob_FP)])
        # Plot the probability of false positives as a function of k
        plt.plot(k_range, prob_FP, label=str(b))
        plt.legend()
    # Add grid lines to the plot
    plt.grid()
    # Add labels to the x-axis and y-axis
    plt.xlabel('Num of K')
    plt.ylabel('Probability')
    plt.title('Bloom filter')
    # Show the plot
    plt.show()
    # Print the optimal k value and the corresponding false positive probability for each b value
    for i in range(len(b_list)):
        print("For b={} The Optimal K is={} and the FP prob is={:.20f}".format(b_list[i], optimal_k[i][0],
                                                                               optimal_k[i][1]))

    # Part 02
    # Create an empty list for storing the theoretical false positive probability
    theory_prob_FP = []
    # Iterate through the b and k values from the calculated optimal k values
    for b, k in zip(b_list, optimal_k):
        # Calculate the theoretical false positive probability using the formula pow(0.5, k[0])
        prob = pow(0.5, k[0])
        # Append the probability, b value and k value to the theory_prob_FP list
        theory_prob_FP.append([prob, b, k[0]])

    # Create a figure for plotting the data
    plt.figure(figsize=(10, 6))
    # Plot the b values against the theoretical false positive probability
    plt.plot([item[1] for item in theory_prob_FP], [item[0] for item in theory_prob_FP])
    # Set the x-axis tick values to be the values in the b_list
    plt.xticks(ticks=b_list)
    # Add scatter points to the plot
    plt.scatter([item[1] for item in theory_prob_FP], [item[0] for item in theory_prob_FP])
    # Add text labels to the scatter points showing the k value
    for item in theory_prob_FP:
        plt.text(item[1], item[0], ' k=' + str(item[2]))
    # Add x and y axis labels
    plt.xlabel('b')
    plt.ylabel('FP Probability')
    # Add a title
    plt.title('FP Probability for Optimal K In Theory (Bloom Filter part 02)')
    # Add grid lines
    plt.grid()
    # Show the plot
    plt.show()

    # Part 04
    # Create an empty list for storing the simulated false positive probability
    prob_simulation = []
    # Create an empty list for storing the memory usage
    memory_simulation = []
    # Iterate through the b and k values from the calculated optimal k values
    for b, k in zip(b_list, optimal_k):
        # n = 2^b, size of the bloom filter
        n = pow(2, b)
        # initialize the bloom filter with size and number of hashes
        BF = Functions.BloomFilter(size=n, hash_count=k[0])
        # add all sentence in sentence_set to the bloom filter
        for sentence in sentence_set:
            BF.add(sentence)
        # count the number of ones in the filter
        onez = BF.count_ones()
        # calculate the false positive probability
        prob = pow(onez / n, k[0])
        prob_simulation.append(prob)
        # calculate the memory occupation
        memory_simulation.append(0.000125 * 1.44 * m * math.log2(1 / prob))

    # creating a figure for the plot
    plt.figure(figsize=(10, 6))
    # plotting the false positive probability against b
    plt.plot(b_list, prob_simulation)
    plt.scatter(b_list, prob_simulation)
    plt.xticks(ticks=b_list)
    # adding k value to each point on the plot
    for b, prob, k in zip(b_list, prob_simulation, optimal_k):
        plt.text(b, prob, ' k=' + str(k[0]))
    plt.xlabel('b')
    plt.ylabel('FP Probability')
    plt.title('FP Probability for Optimal K In simulation(Bloom Filter part 04)')
    plt.grid()
    plt.show()

    # Part 05
    plt.figure(figsize=(10, 6))
    plt.plot([item[1] for item in theory_prob_FP], [item[0] for item in theory_prob_FP], label='theory')
    plt.scatter([item[1] for item in theory_prob_FP], [item[0] for item in theory_prob_FP])
    plt.xticks(ticks=b_list)
    plt.plot(b_list, prob_simulation, label='simulation')
    plt.scatter(b_list, prob_simulation)

    for item in theory_prob_FP:
        plt.text(item[1], item[0], ' k=' + str(item[2]))

    plt.legend()
    plt.xlabel('b')
    plt.ylabel('FP Probability')
    plt.title('Comparison of Theory & Simulation in FP Probability (Bloom Filter Part 05 )')
    plt.grid()
    plt.show()

    # Part 06 (Optional)
    plt.figure(figsize=(12, 8))
    for b, k in zip(b_list, optimal_k):
        elem_counter_list, N_list = [], []
        BF = Functions.BloomFilter(size=2 ** b, hash_count=k[0])
        counter = 1
        f = []
        for sentence in sentence_set:
            BF.add(sentence)
            if counter % 5000 == 0:
                elem_counter_list.append(
                    counter - Functions.distinct_elem_counter(n=2 ** b, k=BF.hash_count, N=BF.count_ones()))
                f.append(counter)
            counter += 1

        plt.plot(f, elem_counter_list, label='b=' + str(b) + ' & k=' + str(k[0]))
        plt.legend()
    plt.xlabel('number of sentences in bloom filter')
    plt.ylabel('difference of actual number of sentences & distinct elements')
    plt.title('Bloom Filter Part 06')
    plt.grid()
    plt.show()
    plt.savefig('optional_part.png')

    # plotting memory occupation in bloom filter simulation
    plt.figure(figsize=(10, 6))
    plt.plot(b_list, memory_simulation)
    plt.scatter(b_list, memory_simulation)
    plt.xlabel('b')
    plt.ylabel("Memory Size (KB)")
    plt.xticks(ticks=b_list)
    plt.title('Memory occupation in Simulation (Bloom Filter)')
    plt.grid()
    plt.show()
