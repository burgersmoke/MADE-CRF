# MADE-CRF
Solution for submitted model for the [UMass MADE 1.0 data challenge](http://bio-nlp.org/index.php/announcements/39-nlp-challenges) by contributors at the University of Utah.

More information on this work can be found in this paper ["Hybrid system for adverse drug event detection"](http://proceedings.mlr.press/v90/chapman18a/chapman18a.pdf)

Which can be cited with the following:
```
@inproceedings{chapman2018hybrid,
  title={Hybrid system for adverse drug event detection},
  author={Chapman, Alec B and Peterson, Kelly S and Alba, Patrick R and DuVall, Scott L and Patterson, Olga V},
  booktitle={International Workshop on Medication and Adverse Drug Event Detection},
  pages={16--24},
  year={2018}
}
```

# Instructions
Best recommendation is to set up a virtual environment to test dependencies in a clean "sandbox" with Anaconda. This project was tested with Anaconda 4.4.0 Python 3.6 (64 bit)

> conda create -n madecrf python=3.6

> activate madecrf

> pip install -r requirements.txt

From this point, everything should be ready to run the notebooks

# Data - MADE data
MADE 1.0 data is to perform feature extraction, training and evaluation.  MADE 1.0 data in the training notebook is expected here:
'C:\MADE' (but easily changed)

Within this directory the following directories are used for the sets released in the MADE 1.0 challenge: 

* MADE-1.0 - Training data
* made1.0_task1_3_test - Dataset released for Task 1 (NER) and 3 (end to end with Relation Extraction) of the challenge
* made_test_data - Final challenge test data

# Data - Embeddings and clusters
To generate clusters from pretrained embeddings, the Wikipedia+PubMed embeddings can be acquired [here](http://evexdb.org/pmresources/vec-space-models/).

Any pretrained embeddings can be found by the embedding clustering notebook at this location (easily changed in the notebook):
'c:\embeddings'

# Notebooks

* CRF_Training.ipynb - Performs training and evaluation on MADE 1.0 data (including feature extraction, CRF training and evaluation)
* Embedding Clustering.ipynb - Performs clustering on pretrained embeddings (whether Wikipedia+PubMed or UPitt embeddings from MADE 1.0)
* Prepare Medex Drug Lexicon.ipynb - Transform MedEx resources into a very simple (non phrasal) lexicon

# Limitations
Note that the best model used pretrained embeddings from UPitt which were made available to MADE 1.0 teams.  

Since it's not clear if these can be distributed, code using these embeddings has been disabled.

# FAQ

* Q : Why are the packages named BASIC?  
* A : Some of these scripts and functions were created on the BASIC (Biosurveillance, Antimicrobial Stewardship and Infection Control) at the University of Utah and Veterans Affairs with Dr. Makoto Jones.

# Troubleshooting

* While loading the POS tagger in Windows (resources/pos/treebank_brill_aubt.pickle) it's possible to see this exception:
	* ModuleNotFoundError: No module named 'copy_reg\r'
	* If so, this can be fixed by using cygwin and running 'dos2unix' to convert this to proper line endings