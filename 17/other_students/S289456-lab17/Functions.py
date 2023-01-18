import hashlib
import re
from math import log as ln
from pympler.asizeof import asizeof
from bitarray import bitarray


class BloomFilter:
    def __init__(self, size: int, hash_count: int):
        # Create a bit array of the desired size
        self.bit_array = bitarray(size)
        # Set all bits to 0
        self.bit_array.setall(0)
        # Store the number of hash functions
        self.hash_count = hash_count

    def add(self, element: str):
        # Hash the element using multiple hash functions
        for i in range(self.hash_count):
            # Calculate the hash value using the modulo method
            element_new = element + str(i)
            word_hash = hashlib.md5(element_new.encode('utf-8'))  # md5 hash
            word_hash_int = int(word_hash.hexdigest(), 16)
            hash_value = word_hash_int % len(self.bit_array)
            # Set the corresponding bit to 1
            self.bit_array[hash_value] = 1

    def count_ones(self) -> int:
        return self.bit_array.count(1)

    def bit_array_size(self):
        return asizeof(self.bit_array)

    def __contains__(self, element: str) -> bool:
        # Hash the element using multiple hash functions
        for i in range(self.hash_count):
            # Calculate the hash value using the modulo method
            element_new = element + str(i)
            word_hash = hashlib.md5(element_new.encode('utf-8'))  # md5 hash
            word_hash_int = int(word_hash.hexdigest(), 16)
            hash_value = word_hash_int % len(self.bit_array)
            # Check if the corresponding bit is set to 1
            if not self.bit_array[hash_value]:
                return False
        return True


def distinct_elem_counter(n: int, k: int, N: int):
    """
    :param n: number of bits
    :param k: number of hash functions
    :param N: is the actual number of bits equal to 1 in the bloom filter
    :return:  the number of distinct elements stored in a bloom filter
    """
    return (-1) * (n / k) * (ln(1 - (N / n)))


def sentence_from_text(sentence_size: int) -> set:
    # reading the La_Divina_Commedia file
    file = open("La_Divina_Commedia.txt", "r").read()
    file = re.sub(r'[^\w\s]', '', file)
    # splitting the text to list of lines
    line_by_line = file.split("\n")
    # removing titles
    del line_by_line[0:8]

    words = []
    # getting the words of the text after removing title of each chapter and spaces before some lines
    for line in line_by_line.copy():

        if line.startswith("Inferno") or line.startswith("Purgatorio") or line.startswith(
                "Paradiso") or line.strip() == "" or line == '':
            line_by_line.remove(line)
        else:
            line = line.strip()
            splited_words = line.split(" ")
            words.extend(splited_words)

    while '' in words:
        words.remove('')

    num_words = len(words)
    dist_num_words = len(set(words))
    print(f'The number words in La Divina Commedia: {num_words}')
    print(f'The number of  distinct words in La Divina Commedia: {dist_num_words}')
    sentences_list = []
    for i in range(len(words)):
        if len(words[i:i + sentence_size]) == sentence_size:
            sentences_list.append(' '.join(words[i:i + sentence_size]))
    return set(sentences_list)


class BitStringArray:
    def __init__(self, size):
        # Create a bit array of the desired size
        self.bit_array = bitarray(size)
        # Set all bits to 0
        self.bit_array.setall(0)

    def bit_array_size(self):
        return asizeof(self.bit_array)

    def hash_function(self, element):
        word_hash = hashlib.md5(element.encode('utf-8'))  # md5 hash
        word_hash_int = int(word_hash.hexdigest(), 16)
        hash_value = word_hash_int % len(self.bit_array)
        return hash_value

    def add(self, element: str):
        hash_value = self.hash_function(element)
        self.bit_array[hash_value] = 1

    def count_ones(self) -> int:
        return self.bit_array.count(1)

    def __contains__(self, element: str) -> bool:
        hash_value = self.hash_function(element)
        # Check if the corresponding bit is set to 1
        if not self.bit_array[hash_value]:
            return False
        else:
            return True


def prob_fp(n: int, m: int) -> float:
    """"
    :param n: number of bits
    :param m: number of sentences
    :return: Probability of FP based on theory formula
    """
    tmp = 1 - (1 / n)
    return 1 - pow(tmp, m)


def generate_fingerprint(sentence: str, n: int):
    # the func to create fingerprint based on the instruction on portal
    sentence_hash = hashlib.md5(sentence.encode('utf-8'))  # md5 hash
    sentence_hash_int = int(sentence_hash.hexdigest(), 16)  # md5 hash in integer format
    h = sentence_hash_int % n  # map into [0,n-1]
    return h
