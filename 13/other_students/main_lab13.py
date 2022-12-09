import re

# First filter
def read_file(name_file='commedia.txt'):
    with open(name_file) as f:
        lines = f.readlines()

    lines = [i.split("\n")[0] for i in lines]
    new_lines = []
    # First cleaning, leading space, title, puntuaction, etc...
    for row in lines:
        if len(row) > 1:
            if (row != ""):
                # remove puntuaction except '
                row = re.sub(r"[^\w\d'\s]+", '', row)
                # Remove more than one space
                row = re.sub(' +', ' ', row)
                # remove starting and closing space
                new_lines.append(row.strip())
    return new_lines[3:]

def distinct_words(list):
    single_words = set()
    for words in list:
        for word in words.split(" "):
            single_words.add(word)
    return len(single_words)

def total_words(list):
    acc = 0
    for words in list:
        acc += len(words.split(" "))
    return acc

if __name__ == '__main__':
    lines = read_file()
    print(f"Total number of words {total_words(lines)}")
    print(f"Number of verses {len(lines)}")
    print(f"Number of distinc words {distinct_words(lines)}")

