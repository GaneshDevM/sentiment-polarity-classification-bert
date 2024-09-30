# Sentiment Polarity Classification Using BERT

## Overview
This project implements a sentiment polarity classification model using the BERT (Bidirectional Encoder Representations from Transformers) architecture. The model classifies movie reviews as either positive or negative based on the Cornell Movie Review Dataset.

## Dataset
The dataset contains a total of 10,662 movie reviews, with 5,331 positive and 5,331 negative samples. The dataset is split into three parts:
- **Training Set**: 4,000 positive and 4,000 negative reviews
- **Validation Set**: 500 positive and 500 negative reviews
- **Test Set**: 831 positive and 831 negative reviews

## Requirements
- Python 3.6+
- PyTorch
- Transformers
- Scikit-learn

## Installation
To set up the project, clone the repository and install the required packages:
```bash
git clone https://github.com/GaneshDevM/sentiment-polarity-classification-bert.git
cd sentiment-polarity-classification-bert
```
To install dataset use this links:
```
https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
```
keep the dataset in the same directory as classifier, then run
```
python sentiment-classifier.py
```
