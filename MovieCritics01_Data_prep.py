from MovieCritics01_functions import *
from collections import Counter

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))

# keep tokens with > 5 occurrence
min_occurrence = 2
tokens = [k for k, c in vocab.items() if c >= min_occurrence]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# prepare negative reviews
negative_lines = process_docs_get_list('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')

# prepare positive reviews
positive_lines = process_docs_get_list('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')