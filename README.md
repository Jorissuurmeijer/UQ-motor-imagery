# Uncertainty Quantification for Motor Imagery BCI - Machine Learning vs. Deep Learning

This repository contains the source code for [Uncertainty Quantification for Motor Imagery BCI - Machine Learning vs. Deep Learning](https://ivopascal.com/uqbci_dl_vs_ml.html), by Joris Suurmeijer, Ivo Pascal de Jong, Matias Valdenegro-Toro and Andreea Ioana Sburlea (2025).

### Abstract
Brain-computer interfaces (BCIs) turn brain signals into functionally useful output, but they are not always accurate.  A good Machine Learning classifier should be able to indicate how confident it is about a given classification, by giving a probability for its classification. 
Standard classifiers for Motor Imagery BCIs do give such probabilities, but research on uncertainty quantification has been limited to Deep Learning. We compare the uncertainty quantification ability of established BCI classifiers using Common Spatial Patterns (CSP-LDA) and Riemannian Geometry (MDRM) to specialized methods in Deep Learning (Deep Ensembles and Direct Uncertainty Quantification) as well as standard Convolutional Neural Networks (CNNs). 

We found that the overconfidence typically seen in Deep Learning is not a problem in CSP-LDA and MDRM. We found that MDRM is underconfident, which we solved by adding Temperature Scaling (MDRM-T). CSP-LDA and MDRM-T give the best uncertainty estimates, but Deep Ensembles and standard CNNs give the best classifications. 
We show that all models are able to separate between easy and difficult estimates, so that we can increase the accuracy of a Motor Imagery BCI by rejecting samples that are ambiguous.


## Install & Run Instructions

In order to copy the code of the experiments run:

```git clone https://github.com/Jorissuurmeijer/UQ-motor-imagery.git```

```cd UQ-motor-imagery```

Then install dependencies with:

```pip install -r requirements.txt```

You can then run the scripts with e.g.:

```python project/models/CSP-LDA/csp_train.py```

The project has different scripts for the different models. The scripts to run the analysis all end in `*_train.py`.
Any necessary data is automatically downloaded through MOABB. The first time you run this it may take some time. 

## Attribution

When using findings or code from this project, please cite the corresponding paper.
  
