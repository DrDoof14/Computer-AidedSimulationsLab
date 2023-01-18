import argparse
import BloomFilter, bit_string_array, fingerprint

parser = argparse.ArgumentParser(description='Lab 17')
parser.add_argument('integers', metavar='N', type=int, choices=[1, 2, 3],
                    help='an integer to trigger a function')

args = parser.parse_args()

if args.integers == 1:
    print('Running the part related to Fingerprint set')
    fingerprint.Fingerprint_part()
elif args.integers == 2:
    print('Running the part related to Bit String Array')
    bit_string_array.bit_string_array()
elif args.integers == 3:
    print('Running the part related to Bloom Filter')
    BloomFilter.bloomfilter()
