#!/usr/bin/env python

"""
Combines an SVM model (SVC or LinearSVC) with a TF-IDF vectorizer to
perform our binary classification task. Different parameter settings can
be specified with command line arguments. Also, optionally, the N most informative
features can be printed.
Authors: Esther Ploeger, Frank van den Berg
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from collections import Counter
import sys
import time
import preprocessing
import pandas as pd


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='data/train',
                        help="Input JSON file(s) to learn from (default 'data/train')")
    parser.add_argument("-t", "--test_file", default='data/COP23.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'data/COP23.filt3.sub.json')")
    parser.add_argument("-m", "--model", default='LinearSVC', type=str,
                        help="The desired SVM model, which is either 'SVC',"
                             " 'LinearSVC' or 'SVR' (default 'LinearSVC')")
    parser.add_argument("-k", "--kernel", default='rbf', type=str,
                        help="The desired kernel when using the SVC or SVR model:"
                             " 'linear' or 'rbf' (default 'rbf')")
    parser.add_argument("-l", "--loss", default='squared_hinge', type=str,
                        help="The desired loss function: "
                             "'hinge' or 'squared_hinge' (default 'squared_hinge')")
    parser.add_argument("-c", "--c_parameter", default=1.0, type=float,
                        help="The value of the SVM's regularization parameter C (default 1.0)")
    parser.add_argument("-n", "--n_gram", default=2, type=int,
                        help="The desired number of N-grams that is given to the TFIDF-"
                             "vectorizer in the parameter ngram_range(1, n) (default=2)")
    parser.add_argument("-min", "--min_df", default=0.01, type=float,
                        help="The desired value for the min_df parameter that is given to the TFIDF-"
                             "vectorizer (default=0.01)")
    parser.add_argument("-max", "--max_df", default=0.32, type=float,
                        help="The desired value for the max_df parameter that is given to the TFIDF-"
                             "vectorizer (default=0.32)")
    parser.add_argument("-mi", "--most_informative", default=0, type=int,
                        help="Print the N most informative features (default 0")

    args = parser.parse_args()
    return args


def identity(x):
    """Dummy function that just returns the input"""
    return x


def choose_model(model, c, loss, kernel):
    """Returns the right model with the specified parameter values"""
    models = {"LinearSVC": LinearSVC(random_state=0, C=c, loss=loss, class_weight='balanced'),
              "SVC": SVC(kernel=kernel, C=c, class_weight='balanced')}

    return models[model]


def print_info(model, kernel, testfile, loss, vectorizer="TF-IDF"):
    """Prints information about the test file, the chosen model and parameter values"""
    print("\nTest file is: {}".format(testfile))
    print("Model: {}".format(model))
    print("Vectorizer: {} vectorizer".format(vectorizer))
    if model == "SVC":
        print("Kernel: {}".format(kernel))
    print("Loss: {}\n".format(loss))


def show_most_informative(feature_names, coefs, n=20):
    """Prints the N most informative features"""
    # Zip coefficients and names together and make a DataFrame
    df = pd.DataFrame(zip(feature_names, coefs), columns=["feature", "value"])
    # Sort the features by the absolute value of their coefficient
    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
    df = df.sort_values("abs_value", ascending=False)

    print(df.head(n))


def create_train_test(train_dir, testfile):
    """Creates train and test data containing articles and labels"""
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

    # Combine vectorizer with the SVM model and print the experiment setup information
    vec = TfidfVectorizer(tokenizer=identity, preprocessor=identity,
                          ngram_range=(1,args.n_gram), min_df=args.min_df, max_df=args.max_df)
    svm_model = choose_model(args.model, args.c_parameter, args.loss, args.kernel)
    clf = Pipeline([('vec', vec), ('cls', svm_model)])
    print_info(args.model, args.kernel, args.test_file, args.loss)

    # Predict the political orientation for the articles
    print("## Predicting...", file=sys.stderr)
    t0 = time.time()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    # Report the precision, recall, f1-score and accuracy scores along with the runtime
    runtime = time.time() - t0
    print(classification_report(Y_test, Y_pred, digits=3))
    print("Completed in {:.1f} seconds\n\n".format(runtime))

    # Optionally, show N most informative features:
    if args.most_informative > 0:
        # Get names and coefficients for each feature, then show most informative features
        feature_names = clf.named_steps["vec"].get_feature_names()
        coefs = clf.named_steps["cls"].coef_.flatten()
        show_most_informative(feature_names, coefs, args.most_informative)
