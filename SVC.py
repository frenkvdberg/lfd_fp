#!/usr/bin/env python

"""
# run as python SVC -i train
(where train is a directory containing COP1-COP22 files)
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVR, SVC
from os import listdir  # to read files
from os.path import isfile, join  # to read files
import matplotlib.pyplot as plt
import sys
import warnings
import os
import json
import re
from collections import Counter
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn import metrics
import time
import preprocessing
import pandas as pd


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='train',
                        help="Input JSON file(s) to learn from (default 'train')")
    parser.add_argument("-t", "--test_file", default='COP23.filt3.sub.json', type=str,
                        help="Input JSON file to test on (default 'COP23.filt3.sub.json')")
    parser.add_argument("-m", "--model", default='LinearSVC', type=str,
                        help="The desired SVM model, which is either 'SVC',"
                             " 'LinearSVC' or 'SVR' (default 'LinearSVC')")
    parser.add_argument("-k", "--kernel", default='rbf', type=str,
                        help="The desired kernel when using the SVC or SVR model:"
                             " 'linear' or 'rbf' (default 'rbf')")
    parser.add_argument("-l", "--loss", default='hinge', type=str,
                        help="The desired loss function: "
                             "'hinge' or 'squared_hinge' (default 'hinge')")
    parser.add_argument("-c", "--c_parameter", default=1.0, type=float,
                        help="The value of the SVM's regularization parameter C (default 1.0)")
    parser.add_argument("-g", "--gamma", default=1.0, type=float,
                        help="The value of the SVM's gamma parameter (default 1.0)")
    parser.add_argument("-n", "--n_gram", default=2, type=int,
                        help="The desired number of N-grams that is given to the tfidf-"
                             "vectorizer in the parameter ngram_range(1, n) (default=2)")
    parser.add_argument("-coef", "--coefficients", action="store_true",
                        help="Print the weights that are assigned to each feature when kernel=linear")

    args = parser.parse_args()
    return args


def identity(x):
    """Dummy function that just returns the input"""
    return x


def choose_model(model, c, loss, kernel, gamma):
    """Returns the right model with the specified parameter values"""
    models = {"LinearSVC": LinearSVC(random_state=0, C=c, loss=loss, class_weight='balanced'),
              "SVR": SVR(kernel=kernel, C=c, gamma=gamma),
              "SVC": SVC(kernel=kernel, C=c, class_weight='balanced')}

    return models[model]


def pos(txt):
    return [p for p in pos_tag(txt)]


def show_most_informative(feature_names, coefs, n=20):
    # Zip coefficients and names together and make a DataFrame
    zipped = zip(feature_names, coefs)
    df = pd.DataFrame(zipped, columns=["feature", "value"])
    # Sort the features by the absolute value of their coefficient
    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
    df = df.sort_values("abs_value", ascending=False)

    print(df.head(n))


if __name__ == "__main__":
    args = create_arg_parser()

    # Get filenames from input (training) directory
    filenames = preprocessing.get_filenames_in_folder(args.input_dir)

    # Create train and test set
    print("## Preprocessing...", file=sys.stderr)
    X_test, Y_test = preprocessing.read_corpus(args.test_file)
    X_train, Y_train = [], []
    for fn in filenames:
        texts, labels = preprocessing.read_corpus(args.input_dir + '/' + fn)
        X_train = X_train + texts
        Y_train = Y_train + labels

    print(Counter(Y_train))
    print(Counter(Y_test))

    c = 1.0
    vec = TfidfVectorizer(tokenizer=identity, preprocessor=identity, ngram_range=(1,2))
    #pos = TfidfVectorizer(tokenizer=pos, preprocessor=identity)
    # union = FeatureUnion([("vec", vec), ("pos", pos)])

    # Get the right model with the specified parameter values
    svm_model = choose_model(args.model, c, args.loss, args.kernel, args.gamma)
    # clf = Pipeline([('union', union), ('cls', svm_model)])
    clf = Pipeline([('vec', vec), ('cls', svm_model)])

    # Predict the tags for the reviews using either cross validation or the regular train and test data:
    t0 = time.time()
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    # Report the precision, recall, f1-score and accuracy scores along with the runtime
    runtime = time.time() - t0
    print(classification_report(Y_test, Y_pred, digits=3))
    print("Completed in {:.1f} seconds".format(runtime))

    # Show N most informative features:
    feature_names = clf.named_steps["vec"].get_feature_names()
    coefs = clf.named_steps["cls"].coef_.flatten()  # Get the coefficients of each feature
    """"
    vf = (clf
                  .named_steps["union"]
                  .transformer_list[0][1]
                  .get_feature_names())
    pf = (clf
               .named_steps["union"]
               .transformer_list[1][1]
               .get_feature_names())
    feature_names = vf + pf
    """
    show_most_informative(feature_names, coefs, 50)
