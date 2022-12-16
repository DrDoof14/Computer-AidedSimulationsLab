import math
import hashlib
from pympler import asizeof


def create_sentence_set(source_file):
    s4 = set() # set to contain sentences with 4 words
    s8 = set() # set to contain sentences with 8 words

    # print the size of the original input file
    print('Total size of original input file:', asizeof.asizeof(open(source_file, 'r', encoding='utf8').read()), 'bytes')

    # read the file and create the sets
    with open(source_file, 'r', encoding='utf8') as f:
        for line in f:
            if len(line.strip().split(' ')) == 4:
                s4.add(line.strip())
            elif len(line.strip().split(' ')) == 8:
                s8.add(line.strip())

    # print the sizes of the sets
    print('Number of sentences with 4 words: ', len(s4))
    print('Number of sentences with 8 words: ', len(s8))

    return s4, s8



def fingerprint(sentence, s):
    word_hash = hashlib.md5(sentence.encode('utf-8')) # generate md5 hash
    word_hash_int = int(word_hash.hexdigest(), 16) # convert md5 hash to integer format
    h = word_hash_int % math.ceil(len(s)/(0.0001)) # map to interval [0,n-1], where n is calculated based on the desired probability of collision (0.0001)

    return h


def create_fingerprint_set(s):
    fingerprint_set = set() # set to contain the fingerprints
    # generate a fingerprint for each sentence and add it to the set
    for sentence in s:
        fingerprint_set.add(fingerprint(sentence, s))
    return fingerprint_set

def check_plagiarism(sentence, s4_fingerprint, s8_fingerprint):
    # check if the sentence is plagiarism
    if len(sentence.strip().split(' ')) == 4:
        # generate a fingerprint for the sentence and check if it is in the set
        return fingerprint(sentence.strip(), s4_fingerprint) in s4_fingerprint
    elif len(sentence.strip().split(' ')) == 8:
        # generate a fingerprint for the sentence and check if it is in the set
        return fingerprint(sentence.strip(), s8_fingerprint) in s8_fingerprint


def main():
    SOURCE_FILE = 'commedia.txt'
    INPUT_SENTENCE = 'del garofano prima discoverse'
    # create sets for sentences with 4 and 8 words
    s4, s8 = create_sentence_set(SOURCE_FILE)
    # generate fingerprints for sentences with 4 and 8 words
    s4_fingerprint = create_fingerprint_set(s4)
    s8_fingerprint = create_fingerprint_set(s8)
    # check if the input sentence is plagiarism
    is_plagiarism = check_plagiarism(INPUT_SENTENCE, s4_fingerprint, s8_fingerprint)
    # print the results
    print('The sentence \'', INPUT_SENTENCE, '\' is' if is_plagiarism else 'is NOT', 'plagiarism')
    print('The ammount of memory used is', asizeof.asizeof(s4_fingerprint) + asizeof.asizeof(s8_fingerprint), 'bytes')
    return is_plagiarism

if __name__ == '__main__':
    main()