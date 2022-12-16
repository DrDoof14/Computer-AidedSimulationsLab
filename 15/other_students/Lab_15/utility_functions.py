import hashlib
import string
from os.path import isfile
from pympler import asizeof
import numpy as np


def read_words(filename):
    words = []
    if not isfile(filename):  # CHECK IF FILE EXISTS
        exit("File not found: " + filename)  # EXIT IF FILE DOES NOT EXIST
    with open(filename, 'r') as f:
        for line in f:
            line = clean_string(line)  # CLEAN LINE
            words += line.split()  # SPLIT LINE INTO WORDS
    return words  # RETURN LIST OF WORDS


def clean_string(s):
    clean_str = s.strip()  # REMOVE SPACES FROM BEGINNING AND END OF STRING
    punctuation = string.punctuation.replace("'", "")  # DEFINE PUNCTUATION
    # REMOVE PUNCTUATION AND CONVERT TO LOWERCASE
    clean_str = clean_str.translate(str.maketrans('', '', punctuation)).lower()
    return clean_str


def get_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        concatenated_str = ' '.join(words[i:i + n])
        ngrams.append(concatenated_str)
    return ngrams


def get_memory_usage(var):
    return asizeof.asizeof(var)


def bytes_to_mb(bytes):
    return round(bytes / 1000000, 3)


def md5_hash(s, n):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % n


def prob_false_positive(n, m):
    return np.ones(len(n)) - ((np.ones(len(n)) - np.ones(len(n)) / n) ** m)


def get_fingerprint_size(m, prob):
    return np.log2((np.ones(len(prob)) * m) / prob)


def plot_fpsize_wrt_probfp(ax, m, window_size):
    x = get_probability_interval()
    y = get_fingerprint_size(m, x)
    ax.plot(x, y, label=f"S = {window_size} (m = {m})", marker='o', markersize=7)
    ax.set_ylabel("Fingerprint size (bit)")  # SET X-AXIS LABEL
    ax.set_xlabel("Probability of false positive")  # SET Y-AXIS LABEL
    ax.set_xscale("log")  # SET X-AXIS SCALE TO LOGARITHMIC


def plot_memory_usage(ax, sentences):
    window_size = len(sentences[0].split())
    m = len(sentences)
    x = get_probability_interval()

    usage_fp_bytes = []
    for eps in x:
        n = m / eps
        fingerprints = set(map(lambda elem: md5_hash(elem, n), sentences))
        usage_fp_bytes.append(bytes_to_mb(get_memory_usage(fingerprints)))

    ax.plot(x, usage_fp_bytes, label=f"S = {window_size} (m = {m})", marker='o', markersize=7)
    ax.set_ylabel("Actual fingerprint size (MB)")  # SET X-AXIS LABEL
    ax.set_xlabel("Probability of false positive")  # SET Y-AXIS LABEL
    ax.set_xscale("log")  # SET X-AXIS SCALE TO LOGARITHMIC


def get_repeated_sentences(listOfElems):
    """ Check if given list contains any duplicates """
    setOfElems = set()
    elems = []
    for elem in listOfElems:
        if elem in setOfElems:
            elems.append(elem)
        else:
            setOfElems.add(elem)
    return elems


def get_probability_interval():
    return [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
