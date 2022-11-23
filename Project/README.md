# Dependency
conda create -n CS5487 python=3.9  
conda activate CS5487  
pip install -r requirement.txt  

# KNN
## no PCA
python machine_learning.py --model kNN 
## PCA
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

# Shallow CNN
## no Aug
python deep_learning.py --model Shallow
## Mixup
python deep_learning.py --model Shallow --aug Mixup

# Deep CNN
## no Aug
python deep_learning.py --model Deep
## Mixup
python deep_learning.py --model Deep --aug Mixup