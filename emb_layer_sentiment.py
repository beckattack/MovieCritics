from numpy import array
import random
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# define documents
docs = ['well done',
		'good work',
		'great effort',
		'nice work',
		'excellent',
		'weak',
		'poor effort',
		'not good',
		'poor work',
		'could have done better']

# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])

#doc test
docs_test = ['not good effort',
		  'excellent',
             'poor']

labels_test = array([0,1,0])

# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print()
print("encode docs:")
print(encoded_docs)

encoded_docs_test = [one_hot(d, vocab_size) for d in docs_test]
print()
print("encode docs_test:")
print(encoded_docs_test)

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print()
print("encoded docs with padding:")
print(padded_docs)

padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
print()
print("encoded docs_test with padding:")
print(padded_docs_test)

random.seed(5)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 4, input_length=max_length))
model.add(LSTM(5))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=80, verbose=1)

embeddings = model.layers[0].get_weights()[0]
print()
print(embeddings)

embedding_for_word_1 = embeddings[1]
print()
print(embedding_for_word_1)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#test the model
labels_pred = model.predict_classes(padded_docs_test)
print(labels_pred)
