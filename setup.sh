#!/bin/bash
# Create directory data/train, in which we will put all training files
mkdir ./data/train
# Move all the COP files to train, except COP23, COP24 and COP25
find data/ -maxdepth 1 -mindepth 1 -not -name COP23.filt3.sub.json -not -name COP24.filt3.sub.json -not -name COP24.filt3.sub.json -type f -print0 | xargs -0 mv -t data/train/
# Unzip the GloVe Embeddings
unzip glove_embeddings300.zip
