#!/usr/bin/env python

"""
Further trains a pre-trained BERT model to
perform our binary classification task. Different parameter settings can
be specified with command line arguments. Also, optionally, the N most informative
features can be printed.
Authors: Esther Ploeger, Frank van den Berg
"""

import sys
import argparse
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import Counter
import nltk
import preprocessing
import evaluate
from pathlib import Path

CACHE_DIR = Path().cwd() / 'cache'
nltk.download('punkt')

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='train',
                        help="Input JSON file(s) to learn from (default 'train')")
    parser.add_argument("-t", "--test_file", default='COP24.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'COP24.filt3.sub.json')")
    parser.add_argument("-d", "--dev_file", default='COP23.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'COP23.filt3.sub.json')")
    parser.add_argument("-ev", "--eval", action="store_true",
                        help="Evaluate the predictions immediately")
    parser.add_argument("-o", "--output_file", default='LM_predictions', type=str,
                        help="The name of the output (pickle) file for the predictions, e.g. 'LM_predictions_test'")
    parser.add_argument('-c', '--cache', default=False, action='store_true',
                        help='Load LM weights from cache file')
    args = parser.parse_args()
    return args


def create_train_test(train_dir, testfile, devfile):
    """Takes all the train files and the test file create
    train and test data containing articles and labels"""
    X_train, Y_train = [], []
    train_filenames = preprocessing.get_filenames_in_folder(train_dir)
    for fn in train_filenames:
        texts, labels = preprocessing.read_corpus(train_dir + '/' + fn)
        X_train = X_train + texts
        Y_train = Y_train + labels
    X_test, Y_test = preprocessing.read_corpus(testfile)
    X_dev, Y_dev = preprocessing.read_corpus(devfile)

    print("# Division of labels\n\tTrain: {}\n\tTest: {}\n\tDev: {}".format(Counter(Y_train), Counter(Y_test), Counter(Y_dev)))
    return X_train, Y_train, X_test, Y_test, X_dev, Y_dev


def create_model():
    """Create the Keras model to use"""
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = 0.00005
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    return model

def train_model(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin):
    """Train the created model using the specified settings"""
    verbose = 1
    # Fit the model to our data
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=1, batch_size=16, validation_data=(tokens_dev, Y_dev_bin))

    # Predict and print evaluation for the model
    return model


def test_set_predict(model, tokens_test):
    """Do predictions and print a classification report"""
    # Get predictions using the trained model
    Y_pred = model.predict(tokens_test)["logits"]
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_pred = ["Right-Center" if pred >= 0.5 else "Left-Center" for pred in Y_pred]
    return Y_pred


if __name__ == "__main__":
    args = create_arg_parser()

    # Create train and test set
    print("## Preprocessing...", file=sys.stderr)
    X_train, Y_train, X_test, Y_test, X_dev, Y_dev = create_train_test(args.input_dir, args.test_file, args.dev_file)

    # Encode textual labels to numerical labels
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)
    Y_test_bin = encoder.fit_transform(Y_test)

    # Create model
    model = create_model()

    # Create BERT input tokens
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens_test = tokenizer(X_test, padding=True, max_length=250, truncation=True, return_tensors="np").data
    tokens_train = tokenizer(X_train, padding=True, max_length=250, truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=250, truncation=True, return_tensors="np").data

    # Either load the saved model from a cache file or create and train it
    cache_file = Path('/seed_1234/bert_weights_1234')
    if args.cache:
        model.load_weights(cache_file)
    else:
        model = train_model(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin)

    # Predict and save predictions
    predictions = test_set_predict(model, tokens_test)
    evaluate.create_prediction_file(args.output_file, predictions)

    # Optionally: print classification report
    if args.eval:
        gold_labels = ["Right-Center" if binary_label == 1 else "Left-Center" for binary_label in Y_test_bin]
        evaluate.print_report(gold_labels, predictions)
