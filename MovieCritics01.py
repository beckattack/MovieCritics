from MovieCritics01_functions import *
from pandas import DataFrame
from matplotlib import pyplot

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset_is_train(vocab, True)
test_docs, ytest = load_clean_dataset_is_train(vocab, False)
# run experiment
modes = ['binary', 'count', 'tfidf', 'freq']
results = DataFrame()
for mode in modes:
    # prepare data for mode
    Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
    # evaluate model on data for mode
    results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
    # summarize results
    print(results.describe())
    # plot results
    results.boxplot()
    pyplot.show()