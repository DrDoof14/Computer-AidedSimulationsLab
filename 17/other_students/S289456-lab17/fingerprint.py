from math import log2
from pympler.asizeof import asizeof
import Functions


# DONE
def Fingerprint_part():
    # getting the set of sentences with the size 6
    sentences_set = Functions.sentence_from_text(sentence_size=6)

    print("The Memory Occupancy of the sentences in a set is: {}KB".format(asizeof(sentences_set) * 0.001))
    # number of sentences
    m = len(sentences_set)
    print("The number of sentences is: {}".format(m))
    print("The Average Size of each sentences in bytes: {:.3f}".format(asizeof(sentences_set) / m))
    Bexp = 0
    # finding the value of the Bexp variable
    for b_i in range(0, 40):
        fingerprint_list = []
        for item in sentences_set:
            h = Functions.generate_fingerprint(sentence=item, n=2 ** b_i)
            fingerprint_list.append(h)
        # condition for no collision
        if len(fingerprint_list) - len(set(fingerprint_list)) == 0:
            Bexp = b_i
            print('Bexp is Equal to: ', Bexp)
            break
    # finding the value of Bteo by using the formula
    Bteo = round(log2(pow(m / 1.17, 2)))
    print('Bteo is Equal to: ', Bteo)

    Pr_FP_Bexp = 1 - pow((1 - 1 / pow(2, Bexp)), m)
    print('The Probability of False Positive for Bexp is: {:.20f}'.format(Pr_FP_Bexp))
