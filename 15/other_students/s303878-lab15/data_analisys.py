import math
import matplotlib.pyplot as plt
import hashlib
from pympler import asizeof

def length_of_fingerprint(m4, m8):
    results4 = []
    results8 = []
    # calculate the length in bytes of the fingerprint for each probability of collision between 0.001 and 1
    for e in range(1, 100):
        results4.append(math.ceil(math.log(m4/(e/100), 2)))
        results8.append(math.ceil(math.log(m8/(e/100), 2)))
    
    # plot the results
    # create a figure with two subplots side by side
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # plot the results for 4 words sentences
    ax1.plot(range(1, 100), results4)
    ax1.set_title('4 words sentences')
    ax1.set_xlabel('Probability of collision')
    ax1.set_ylabel('Length of fingerprint (bytes)')
    # plot the results for 8 words sentences
    ax2.plot(range(1, 100), results8)
    ax2.set_title('8 words sentences')
    ax2.set_xlabel('Probability of collision')
    ax2.set_ylabel('Length of fingerprint (bytes)')
    plt.show()


def fingerprint(sentence, s, e):
    word_hash = hashlib.md5(sentence.encode('utf-8')) # generate md5 hash
    word_hash_int = int(word_hash.hexdigest(), 16) # convert md5 hash to integer format
    h = word_hash_int % math.ceil(len(s)/e) # map to interval [0,n-1], where n is calculated based on the desired probability of collision e

    return h


def create_fingerprint_set(s, e):
    fingerprint_set = set() # set to contain the fingerprints
    # generate a fingerprint for each sentence and add it to the set
    for sentence in s:
        fingerprint_set.add(fingerprint(sentence, s, e))
    return fingerprint_set


def size_in_memory(s4, s8):
    s4_fingerprint_set_list = []
    s8_fingerprint_set_list = []

    # create fingerprint sets for each probability of collision between 0.001 and 1
    for e in range(1, 100):
        s4_fingerprint_set_list.append(create_fingerprint_set(s4, e))
        s8_fingerprint_set_list.append(create_fingerprint_set(s8, e))

    # plot the results
    # create a figure with two subplots side by side
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # plot the results for 4 words sentences
    ax1.plot(range(1, 100), [asizeof.asizeof(s4_fingerprint) for s4_fingerprint in s4_fingerprint_set_list])
    ax1.set_title('4 words sentences')
    ax1.set_xlabel('Probability of collision')
    ax1.set_ylabel('Size of fingerprint set (bytes)')
    # plot the results for 8 words sentences
    ax2.plot(range(1, 100), [asizeof.asizeof(s8_fingerprint) for s8_fingerprint in s8_fingerprint_set_list])
    ax2.set_title('8 words sentences')
    ax2.set_xlabel('Probability of collision')
    ax2.set_ylabel('Size of fingerprint set (bytes)')
    plt.show()


def create_sentence_set(source_file):
    s4 = set()
    s8 = set()

    with open(source_file, 'r') as f:
        for line in f:
            if len(line.strip().split(' ')) == 4:
                s4.add(line.strip())
            elif len(line.strip().split(' ')) == 8:
                s8.add(line.strip())
    return s4, s8


def main():
    SOURCE_FILE = 'commedia.txt'
    # create sets for sentences with 4 and 8 words
    s4, s8 = create_sentence_set(SOURCE_FILE)
    # calculate results and plot them
    length_of_fingerprint(len(s4), len(s8))
    size_in_memory(s4, s8)


if __name__ == '__main__':
    main()