#!/usr/bin/env python

"""
Combines one of the specified classic models (e.g. Naive Bayes) with
either a count vectorizer or a TF-IDF vectorizer to perform our
binary classification task. Optionally, a confusion matrix can be printed.
Authors: Esther Ploeger, Frank van den Berg
"""


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import sys
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

import preprocessing


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='data/train',
                        help="Input JSON file(s) to learn from (default 'data/train')")
    parser.add_argument("-t", "--test_file", default='data/COP23.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'data/COP23.filt3.sub.json')")
    parser.add_argument("-m", "--model", default='NB', type=str,
                        help="Classifier model, which can be 'NB' (Naive Bayes), 'DT' (Decision Tree), "
                             "'RF' (Random Forest), 'MF' (Always predicting most frequent class), "
                             "'LinearSVC', or 'SVC'. Default choice is 'NB'")
    parser.add_argument("-k", "--kernel", default='linear', type=str,
                        help="Kernel when using the SVC classifier. Either 'linear' or 'rbf' (default 'linear')")
    parser.add_argument("-c", "--criterion", default='gini', type=str,
                        help="Value of criterion parameter when using Decision Tree or Random Forest. "
                        "Either gini' or 'entropy' (default 'gini')")
    parser.add_argument("-v", "--vectorizer", default='count', type=str,
                        help="Vectorizer to combine with the classifier. "
                             "Either 'count' or 'TFIDF' (default 'count')")
    parser.add_argument("-cm", "--confusion_matrix", action="store_true",
                        help="Print a confusion matrix")
    args = parser.parse_args()
    return args


def print_cm(Y_test, Y_pred):
    """Generates a confusion matrix and prints the results along with the labels"""
    labels = list(set(Y_test))
    abbreviations = [l[:3] for l in labels]
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)

    # Print the matrix with corresponding labels above and next to each result
    print("\n## Confusion matrix (Gold vertical vs Predicted horizontal):")
    print("{:<4} {}".format("", " ".join(abbreviations)))
    for l in cm:
        print("{} {}".format(abbreviations.pop(0), l))


def identity(x):
    """Dummy function that just returns the input"""
    return x


def choose_model(model, kernel, criterion):
    """Returns the right model with the specified parameter values"""
    models = {"LinearSVC": LinearSVC(random_state=0, class_weight='balanced', loss='hinge'),
              "SVC": SVC(random_state=0, class_weight='balanced', kernel=kernel),
              "DT": DecisionTreeClassifier(random_state=0, class_weight="balanced", criterion=criterion),
              "RF": RandomForestClassifier(random_state=0, class_weight="balanced", criterion=criterion),
              "NB": MultinomialNB(),
              "MF": DummyClassifier(strategy='most_frequent')}

    return models[model]


def print_info(model, kernel, criterion, vectorizer, testfile):
    """Prints information about the test file, the chosen model and parameter values"""
    print("\nTest file is: {}".format(testfile))
    print("\nModel: {}".format(model))
    print("Vectorizer: {} vectorizer".format(vectorizer))
    if model == "SVC":
        print("Kernel: {}".format(kernel))
    elif model == "DT" or model == "RF":
        print("Criterion: {}".format(criterion))


def create_train_test(train_dir, testfile):
    """Takes all the train files and the test file create
    train and test data containing articles and labels"""
    X_train, Y_train = [], []
    train_filenames = preprocessing.get_filenames_in_folder(train_dir)
    for fn in train_filenames:
        texts, labels = preprocessing.read_corpus(train_dir + '/' + fn)
        X_train = X_train + texts
        Y_train = Y_train + labels
    X_test, Y_test = preprocessing.read_corpus(testfile)

    print("# Division of labels\n\tTrain: {}\n\tTest: {}".format(Counter(Y_train), Counter(Y_test)))
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    args = create_arg_parser()

    # Create train and test set
    print("## Preprocessing...", file=sys.stderr)
    X_train, Y_train, X_test, Y_test = create_train_test(args.input_dir, args.test_file)

    # Define vectorizer and model, then combine in a pipeline:
    if args.vectorizer == "TFIDF":
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
    model = choose_model(args.model, args.kernel, args.criterion)
    classifier = Pipeline([('vec', vec), ('cls', model)])

    # Fit the classifier on the training data, predict the output
    # and print the classification report
    print_info(args.model, args.kernel, args.criterion, args.vectorizer, args.test_file)
    t0 = time.time()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    print("Completed in {:.1f} seconds\n".format(time.time() - t0))
    print(classification_report(Y_test, Y_pred, digits=3, zero_division=0))

    # Optional: print a confusion matrix
    if args.confusion_matrix:
        print_cm(Y_test, Y_pred)

