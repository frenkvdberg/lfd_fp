# lfd_fp

### Retrieving all the necessary files and setting up
Before we can run the models, we first need to make sure that we have access to all the files by following these steps:
1. Download the trained lstm model as a .zip file from dropbox using this link:
https://www.dropbox.com/s/cojnks0t2w15vbp/lstm_model.zip?dl=0
Then extract the zip file into the cache directory
2. DOWNLOAD BERT MODEL -- HIER LINK
3. Run the setup.sh shell script that will create a 'train' directory containing the COP files from 1 to 22, it also extracts the GloVe embeddings:
```bash
$ code to run shellscript
``` 
4. Install all dependencies by running the following command:
```bash
$ pip install -r requirements.txt
``` 


### Training the models and testing on unseen data
Vertellen hoe de verschillende modellen gebruikt kunnen worden om te trainen en te testen, waarbij een output file wordt aangemaakt.
En dat er met de optie -ev meteen een classification report geprint kan worden, maar dat een evaluatie ook los kan door evaluate.py te gebruiken. 
