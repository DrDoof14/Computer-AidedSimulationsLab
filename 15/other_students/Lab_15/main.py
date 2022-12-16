from utility_functions import read_words, get_ngrams, get_memory_usage, md5_hash, bytes_to_mb, plot_fpsize_wrt_probfp, \
    plot_memory_usage
from matplotlib import pyplot as plt

POEM_FILENAME = "commedia.txt"
WINDOW_SIZES = [4, 8, 16]


def main():
    words = read_words(POEM_FILENAME)

    fig1 = plt.figure(1, (12, 7))
    fig2 = plt.figure(2, (12, 7))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    for window_size in WINDOW_SIZES:
        sentences = get_ngrams(words, window_size)
        distinct_sentences = set(sentences)
        usage_dist_str_bytes = get_memory_usage(distinct_sentences)
        usage_str_bytes = get_memory_usage(sentences)

        print(f"WINDOW SIZE: {str(window_size):2}")
        print("# Total Sentences: {:23}".format(len(sentences)))
        print("# Distinct sentences: {:20}".format(len(distinct_sentences)))
        print("Total strings memory usage: {:14} bytes ({:6} MB)".format(usage_str_bytes, bytes_to_mb(usage_str_bytes)))
        print("Distinct strings memory usage: {:11} bytes ({:6} MB)".format(usage_dist_str_bytes,
                                                                            bytes_to_mb(usage_dist_str_bytes)))
        print("")
        plot_fpsize_wrt_probfp(ax1, len(distinct_sentences), window_size)
        plot_memory_usage(ax2, list(distinct_sentences))

    ax1.set_title("Fingerprint size in function of P(FP) = log2(m / P(FP))")
    ax2.set_title("Actual fingerprint memory usage in function of P(FP)")
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.show()  # SHOW PLOT


if __name__ == '__main__':
    main()
