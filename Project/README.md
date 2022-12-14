# Programming Project -- MNIST Classification

This repo provides an implementation of CS5487 Programming Project (City University of Hong Kong) -- MNIST Classification

## Datasets
The datasets are provided by course (./digits4000_txt and ./challenge).

## Experimental Dependency (python)
```shell
# create virtual environment
conda create --name=CS5487 python=3.9

# activate virtual environment
conda activate CS5487

# install dependencies
pip install -r requirements.txt
```

## Usage
```shell
# create files for models
mkdir models/

# KNN
## no PCA
python machine_learning.py --model kNN 
## with PCA
python machine_learning.py --model kNN --PCA 1

# LR
## no PCA
python machine_learning.py --model LR 
## PCA
python machine_learning.py --model LR --PCA 1

# Perceptron
## no PCA
python machine_learning.py --model Per 
## PCA
python machine_learning.py --model Per --PCA 1

# SVM
## no PCA
python machine_learning.py --model SVM
## PCA
python machine_learning.py --model SVM --PCA 1

# MLP
## no Aug
python deep_learning.py --model MLP
## Mixup
python deep_learning.py --model MLP --aug Mixup

# CNN
## no Aug
python deep_learning.py --model CNN
## Mixup
python deep_learning.py --model CNN --aug Mixup

# Challenge
python challenge_test.py
```
