#!/usr/bin/env python

"""
Create and train a LSTM model with pretrained GloVe embeddings to
perform our binary classification task. Different parameter settings can
be specified with command line arguments.
Authors: Esther Ploeger, Frank van den Berg
"""

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM
from keras.initializers import Constant
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import preprocessing
from keras.models import load_model
from pathlib import Path
from collections import Counter
import evaluate

CACHE_DIR = Path().cwd() / 'cache'


# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)




def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='data/train',
                        help="Input JSON file(s) to learn from (default 'data/train')")
    parser.add_argument("-t", "--test_file", default='None', type=str,
                        help="Input JSON file to test on, e.g. 'data/COP24.filt3.sub.json' (default 'None')")
    parser.add_argument("-d", "--dev_file", default='data/COP23.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'data/COP23.filt3.sub.json')")
    parser.add_argument("-e", "--embeddings", default='glove_embeddings300.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    parser.add_argument("-ep", "--epochs", default='22', type=int,
                        help="The number of epochs to use in training the model (default 22)")
    parser.add_argument("-lr", "--learning_rate", default='0.0005', type=float,
                        help="The learning rate to use in training the model (default 0.0005)")
    parser.add_argument("-b", "--batch_size", default='16', type=int,
                        help="The batch size to use in training the model (default 16)")
    parser.add_argument("-op", "--optimizer", default='Nadam', type=str,
                        help="The optimizer to use in training the model, either SGD, "
                             "Adam or Nadam (default Nadam)")
    parser.add_argument('-c', '--cache', default=False, action='store_true',
                        help='Load LSTM model from cache file')
    parser.add_argument("-o", "--output_file", default='LSTM_predictions', type=str,
                        help="The name of the output (pickle) file for the predictions, e.g. 'LSTM_predictions_dev'")
    parser.add_argument("-ev", "--eval", action="store_true",
                        help="Evaluate the predictions immediately")
    args = parser.parse_args()
    return args


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def create_train_test(train_dir, testfile):
    """Takes all the train files and the test file to create
    train and test data containing articles and labels"""
    X_train, Y_train = [], []
    train_filenames = preprocessing.get_filenames_in_folder(train_dir)
    for fn in train_filenames:
        texts, labels = preprocessing.read_corpus(train_dir + '/' + fn, "sent")
        X_train = X_train + texts
        Y_train = Y_train + labels
    X_test, Y_test = preprocessing.read_corpus(testfile, "sent")

    print("# Division of labels\n\tTrain: {}\n\tTest: {}".format(Counter(Y_train), Counter(Y_test)))
    return X_train, Y_train, X_test, Y_test


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Using a little trick, we get the embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def get_optimizer(optimizer, lr):
    """Get optimizer with the specified learning rate"""
    optimizers = {"adam": Adam(learning_rate=lr),
                  "nadam": Nadam(learning_rate=lr),
                  "sgd": SGD(learning_rate=lr)}
    return optimizers[optimizer.lower()]


def create_model(emb_matrix, learning_rate, optimizer):
    """Create the Keras model to use"""
    loss_function = 'binary_crossentropy'
    optim = get_optimizer(optimizer, learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))
    model.add(LSTM(units=128, activation='sigmoid'))
    model.add(Dense(input_dim=embedding_dim, units=1, activation="sigmoid"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, bs, ep):
    """Train the created model using the specified settings"""
    verbose = 1
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=ep, batch_size=bs, validation_data=(X_dev, Y_dev))
    model.save(Path(CACHE_DIR / 'lstm_model.h5'))  # Save the model to disk
    return model


def test_set_predict(model, X_test):
    """Do predictions and convert them to the right labels"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Convert to PO labels
    Y_pred = ["Right-Center" if pred >= 0.5 else "Left-Center" for pred in Y_pred]
    return Y_pred



def main():
    """Main function to train and test neural network given cmd line arguments"""
    args = create_arg_parser()

    # Create train set, dev set and embeddings
    X_train, Y_train, X_dev, Y_dev = create_train_test(args.input_dir, args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer and use train to create vocab
    vectorizer = TextVectorization(standardize=None, output_sequence_length=250)
    text_ds = tf.data.Dataset.from_tensor_slices(X_train)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings and transform input to vectorized input
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Either load the saved model from a cache file or create and train it
    cache_file = Path(CACHE_DIR / 'lstm_model.h5')
    if args.cache and cache_file.is_file():
        model = load_model(cache_file)
    else:
        model = create_model(emb_matrix, args.learning_rate, args.optimizer)
        model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, args.batch_size, args.epochs)

    # Predict on testfile if filename is specified, otherwise on dev
    if args.test_file != 'None':
        # Read in test set, vectorize and predict
        X_test, Y_test = preprocessing.read_corpus(args.test_file, "sent")
        X_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        Y_bin = encoder.fit_transform(Y_test)
    else:
        X_vect = X_dev_vect
        Y_bin = Y_dev_bin

    # Predict and save predictions
    predictions = test_set_predict(model, X_vect)
    evaluate.create_prediction_file(args.output_file, predictions)

    # Optionally: print classification report
    if args.eval:
        gold_labels = ["Right-Center" if binary_label == 1 else "Left-Center" for binary_label in Y_bin]
        evaluate.print_report(gold_labels, predictions)


if __name__ == '__main__':
    main()
