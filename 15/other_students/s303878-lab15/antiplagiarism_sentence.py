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


def check_plagiarism(sentence, s4, s8):
    # check if the sentence is plagiarism
    if len(sentence.strip().split(' ')) == 4:
        # check if the sentence is in the set
        return sentence.strip() in s4
    elif len(sentence.strip().split(' ')) == 8:
        # check if the sentence is in the set
        return sentence.strip() in s8


def main():
    SOURCE_FILE = 'commedia.txt'
    INPUT_SENTENCE = 'del garofano prima discoverse'
    # create sets for sentences with 4 and 8 words
    s4, s8 = create_sentence_set(SOURCE_FILE)
    # check if the input sentence is plagiarism
    is_plagiarism = check_plagiarism(INPUT_SENTENCE, s4, s8)
    # print the results
    print('The sentence \'', INPUT_SENTENCE, '\' is' if is_plagiarism else 'is NOT', 'plagiarism')
    print('The ammount of memory used is', asizeof.asizeof(s4) + asizeof.asizeof(s8), 'bytes')
    return is_plagiarism

if __name__ == '__main__':
    main()