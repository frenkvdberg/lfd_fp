#!/usr/bin/env python

"""
Reads the predictions from a pickle file, takes the gold labels
from the specified JSON file and prints a classification report.
Authors: Esther Ploeger, Frank van den Berg
"""

import argparse
import pickle
from sklearn.metrics import classification_report
from pathlib import Path
import preprocessing
from sklearn.metrics import confusion_matrix

OUTPUT_DIR = Path().cwd() / 'output'


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='output/NB_predictions',
                        help="Input pickle file to get predictions from (default 'output/NB_predictions')")
    parser.add_argument("-t", "--test_file", default='data/COP23.filt3.sub.json', type=str,
                        help="JSON file to get gold labels from (default 'data/COP23.filt3.sub.json')")
    parser.add_argument("-cm", "--confusion_matrix", action="store_true",
                        help="Print a confusion matrix")
    args = parser.parse_args()
    return args


def print_report(predictions, gold_labels):
    """Print a classification report"""
    print(classification_report(gold_labels, predictions, digits=3, zero_division=0))


def create_prediction_file(filename, predictions):
    """Writes predictions to output pickle file to use for later evaluation"""
    with open(Path(OUTPUT_DIR / filename), 'wb') as f:
        pickle.dump(predictions, f)


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


if __name__ == "__main__":
    args = create_arg_parser()

    # Read the predictions from a pickle file
    with open(args.input_file, 'rb') as f:
        predictions_list = pickle.load(f)

    # Get the gold labels from the specified test file
    X_test, Y_test = preprocessing.read_corpus(args.test_file)

    # Print classification report
    print_report(predictions_list, Y_test)

    # Optional: print a confusion matrix
    if args.confusion_matrix:
        print_cm(Y_test, predictions_list)