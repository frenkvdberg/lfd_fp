# Predicting the source of Australian climate-related newspaper articles
## Learning from Data 2021-2022: Final Project

### Retrieving all the necessary files and setting up
Before we can run the models, we first need to make sure that we have access to all the files by following these steps:
1. Download the trained lstm model as a .zip file from dropbox using this link:<br />
https://www.dropbox.com/s/cojnks0t2w15vbp/lstm_model.zip?dl=0<br />
Then extract 'lstm_model.h5' from the zip file and put it into the cache directory (which was empty before) <br />
2. Download the lm weights as a .zip file from google drive using this link:<br />
https://drive.google.com/drive/folders/10TzwInvX-22RjvXH56fHWXo0Y2xT9mHY?usp=sharing
Then extract the directory and put it into the cache directory <br />
4. Run the setup.sh shell script that will create a 'train' directory* containing the COP files from 1 to 22, it also extracts the GloVe embeddings:
```bash
chmod +x ./setup.sh
sh ./setup.sh
``` 
\* note that for this script to work, we expect the all the COP files to be in the directory 'data', since we can not upload it ourselves.

4. Install all dependencies by running the following command:
```bash
pip install -r requirements.txt
``` 
<hr style="border:2px solid gray"> </hr>

### Training the models and testing on unseen data
Training the model and making the predictions can be done using the follow commands for each model:

#### NB
```bash
python NB.py -t data/COP24.filt3.sub.json -o NB_test -ev
```
For each model, we can use the -t parameter to specify the file that we want to test on;<br />
The -o parameter is used to specify the filename for the output (pickle-)file that the predictions are saved in;<br />
The -ev parameter gives us the option to print a classifcation report right away (While we can also evaluate output files using evaluate.py, which we will discuss later).<br />
A list of all the possible command line arguments can be requested with the -h option.

#### SVM
```bash
python SVM.py -t data/COP24.filt3.sub.json -o SVM_test -ev
```
Note that we do not use a pretrained model for the SVM, since it takes our best model only 11 seconds to predict on unseen data.<br />

#### LSTM
```bash
python LSTM.py -c -t data/COP24.filt3.sub.json -o LSTM_test -ev
```
Here we use the -c parameter to specify that we want to use the trained model that is stored in the cache directory. If there is no model saved in this directory, the model will train normally and after that the model will be saved automatically into this directory.

#### BERT
```bash
python bert.py -c -t data/COP24.filt3.sub.json -o LM_test -ev
```
Here, we use the -c parameter to specify that we want to use the weights that are stored in the cache directory.

<hr style="border:2px solid gray"> </hr>

### Evaluating output files
While each model can print a classification report when we use the -ev option, we can also use evaluate.py to run the evaluation of a given output file:

```bash
python evaluate.py -i output/SVMtest -t data/COP24.filt3.sub.json -cm
```

With -i we can specify which output file we want to evaluate. Each output file is a pickle file containing a list of predicted labels.<br />
With -t we specify which test file is used to obtain the gold labels;<br />
The -cm option can be used to print a confusion matrix.
