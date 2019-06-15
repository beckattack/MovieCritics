#Example source: see https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb

import gzip
import gensim
import logging
import random
from gensim.test.utils import datapath

train = 0
if train == 0:
            data_file = "./reviews_data.txt.gz"
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            with gzip.open('reviews_data.txt.gz', 'rb') as f:
                for i, line in enumerate(f):
                    print(line)
                    break


            def read_input(input_file):
                """This method reads the input file which is in gzip format"""

                logging.info("reading file {0}...this may take a while".format(input_file))

                with gzip.open(input_file, 'rb') as f:
                    for i, line in enumerate(f):

                        if (i % 10000 == 0):
                            logging.info("read {0} reviews".format(i))
                        # do some pre-processing and return a list of words for each review text
                        yield gensim.utils.simple_preprocess(line)


            # read the tokenized reviews into a list
            # each review item becomes a series of words
            # so this becomes a list of lists


            documents = list(read_input(data_file))
            logging.info("Done reading data file")

            model = gensim.models.Word2Vec (documents, size=50, window=10, min_count=2, workers=4)
            model.train(documents,total_examples=len(documents), epochs=10)
            word_vectors = model.wv
            word_vectors.save("word2vec.model")



model = gensim.models.keyedvectors.KeyedVectors.load("./word2vec.model")

# Get numpy vector of word
print()
print(model.wv['hotel'])


w1 = "dirty"
print()
print(model.wv.most_similar (positive=w1))

# look up top 3 words similar to 'polite'
w1 = ["polite"]
print()
print(model.wv.most_similar (positive=w1, topn=3))


# look up top 6 words similar to 'shocked'
w1 = ["shocked"]
print()
print(model.wv.most_similar (positive=w1, topn=6))



