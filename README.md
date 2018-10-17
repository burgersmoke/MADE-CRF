# MADE-CRF
Solution for submitted model for the UMass MADE 1.0 data challenge

# Instructions

Best recommendation is to set up a virtual environment to test dependencies in a clean "sandbox" with Anaconda

This project was tested with Anaconda 4.4.0 Python 3.6 (64 bit)

> conda create -n madecrf python=3.6
> activate madecrf
> pip install -r requirements.txt

From this point, everything should be ready to run the notebooks

# Notebooks

* CRF_Training.ipynb - Performs training and evaluation on MADE 1.0 data (including feature extraction, CRF training and evaluation)
* Embedding Clustering.ipynb - Performs clustering on pretrained embeddings (whether Wikipedia+PubMed or UPitt embeddings from MADE 1.0)
* Prepare Medex Drug Lexicon.ipynb - Transform MedEx resources into a very simple (non phrasal) lexicon