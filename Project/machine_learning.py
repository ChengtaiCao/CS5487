import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils import *


def kNN_function(data_dict, PCA_flag, str_txt, n_splits=10):
    """
    kNN
    Input:
        data_dict: data dict
        PCA_flag: PCA or not
        str_txt: print text
        n_splits: number of cross validation
    """
    trail_1, trail_2 = pre_process_ML(data_dict)
    Ks = []
    PCAs = []
    scores = []
    for data in [trail_1, trail_2]:
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        # grid_search
            # k in [1, 2, 3, ..., 20]
            # pca [0.1, 0.2, ..., 0.9]
        if PCA_flag:
            pipe = Pipeline([
                ('pca', PCA(svd_solver='full')),
                ('clf', KNeighborsClassifier()),
            ])
            parameters = {
                'pca__n_components': [i/100 for i in range(10, 100, 10)],
                'clf__n_neighbors': np.arange(1, 21),
            }
        else:
            pipe = Pipeline([
                ('clf', KNeighborsClassifier()),
            ])
            parameters = {
                'clf__n_neighbors': np.arange(1, 21),
            }
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, parameters, cv=kf, n_jobs=-1, verbose=1)
        gs.fit(train_x, train_y)

        if PCA_flag:
            best_K = gs.best_estimator_.get_params()["clf__n_neighbors"]
            best_PCA_component = gs.best_estimator_.get_params()["pca__n_components"]
            pipe = Pipeline([
                ('pca', PCA(n_components=best_PCA_component, svd_solver='full')),
                ('clf', KNeighborsClassifier(n_neighbors=best_K)),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            Ks.append(best_K)
            PCAs.append(best_PCA_component)
            scores.append(score)
        else:
            best_K = gs.best_estimator_.get_params()["clf__n_neighbors"]
            pipe = Pipeline([
                ('clf', KNeighborsClassifier(n_neighbors=best_K)),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            Ks.append(best_K)
            scores.append(score)

    if PCA_flag:
        print(f"kNN + {str_txt}")
        print(f"Two trails for best Ks are {Ks}")
        print(f"Two trails for best PCA components are {PCAs}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")
    else:
        print(f"kNN + {str_txt}")
        print(f"Two trails for best Ks are {Ks}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")


def LR_function(data_dict, PCA_flag, str_txt, n_splits=10):
    """
    Logistic Regression
    Input:
        data_dict: data dict
        PCA_flag: PCA or not
        str_txt: print text
        n_splits: number of cross validation
    """
    trail_1, trail_2 = pre_process_ML(data_dict)
    PCAs = []
    scores = []
    for data in [trail_1, trail_2]:
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        # grid_search
            # pca [0.1, 0.2, ..., 0.9]
        if PCA_flag:
            pipe = Pipeline([
                ('pca', PCA(svd_solver='full')),
                ('lr', LogisticRegression()),
            ])
            parameters = {
                'pca__n_components': [i/100 for i in range(10, 100, 10)],
            }    

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe, parameters, cv=kf, n_jobs=-1, verbose=1)
            gs.fit(train_x, train_y)

            best_PCA_component = gs.best_estimator_.get_params()["pca__n_components"]
            pipe = Pipeline([
                ('pca', PCA(n_components=best_PCA_component, svd_solver='full')),
                ('lr', LogisticRegression()),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            PCAs.append(best_PCA_component)
            scores.append(score)
        else:
            pipe = Pipeline([
                ('lr', LogisticRegression()),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            scores.append(score)

    if PCA_flag:
        print(f"Logistic Regression + {str_txt}")
        print(f"Two trails for best PCA components are {PCAs}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")
    else:
        print(f"Logistic Regression + {str_txt}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")


def Perceptron_function(data_dict, PCA_flag, str_txt, n_splits=10):
    """
    Perceptron
    Input:
        data_dict: data dict
        PCA_flag: PCA or not
        str_txt: print text
        n_splits: number of cross validation
    """
    trail_1, trail_2 = pre_process_ML(data_dict)
    PCAs = []
    scores = []
    for data in [trail_1, trail_2]:
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        # grid_search
            # pca [0.1, 0.2, ..., 0.9]
        if PCA_flag:
            pipe = Pipeline([
                ('pca', PCA(svd_solver='full')),
                ('per', Perceptron()),
            ])
            parameters = {
                'pca__n_components': [i/100 for i in range(10, 100, 10)],
            }    

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe, parameters, cv=kf, n_jobs=-1, verbose=1)
            gs.fit(train_x, train_y)

            best_PCA_component = gs.best_estimator_.get_params()["pca__n_components"]
            pipe = Pipeline([
                ('pca', PCA(n_components=best_PCA_component, svd_solver='full')),
                ('per', Perceptron()),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            PCAs.append(best_PCA_component)
            scores.append(score)
        else:
            pipe = Pipeline([
                ('per', Perceptron()),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            scores.append(score)

    if PCA_flag:
        print(f"Perceptron + {str_txt}")
        print(f"Two trails for best PCA components are {PCAs}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")
    else:
        print(f"Perceptron + {str_txt}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")


def SVM_function(data_dict, PCA_flag, str_txt, n_splits=10):
    """
    SVM
    Input:
        data_dict: data dict
        PCA_flag: PCA or not
        str_txt: print text
        n_splits: number of cross validation
    """
    trail_1, trail_2 = pre_process_ML(data_dict)
    best_kernels = []
    PCAs = []
    scores = []
    for data in [trail_1, trail_2]:
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        # grid_search
            # k in [1, 2, 3, ..., 20]
            # pca [0.1, 0.2, ..., 0.9]
        if PCA_flag:
            pipe = Pipeline([
                ('pca', PCA(svd_solver='full')),
                ('svm', SVC()),
            ])
            parameters = {
                'pca__n_components': [i/100 for i in range(10, 100, 10)],
                'svm__kernel': ["linear", "poly", "rbf", "sigmoid"]
            }
        else:
            pipe = Pipeline([
                ('svm', SVC()),
            ])
            parameters = {
                'svm__kernel': ["linear", "poly", "rbf", "sigmoid"]
            }
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, parameters, cv=kf, n_jobs=-1, verbose=1)
        gs.fit(train_x, train_y)

        if PCA_flag:
            best_kernel = gs.best_estimator_.get_params()["svm__kernel"]
            best_PCA_component = gs.best_estimator_.get_params()["pca__n_components"]
            pipe = Pipeline([
                ('pca', PCA(n_components=best_PCA_component, svd_solver='full')),
                ('svm', SVC(kernel=best_kernel)),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            best_kernels.append(best_kernel)
            PCAs.append(best_PCA_component)
            scores.append(score)
        else:
            best_kernel = gs.best_estimator_.get_params()["svm__kernel"]
            pipe = Pipeline([
                ('clf', SVC(kernel=best_kernel)),
            ])
            pipe.fit(train_x, train_y)
            score = pipe.score(test_x, test_y)
            best_kernels.append(best_kernel)
            scores.append(score)

    if PCA_flag:
        print(f"SVM + {str_txt}")
        print(f"Two trails for best kernels are {best_kernels}")
        print(f"Two trails for best PCA components are {PCAs}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")
    else:
        print(f"SVM + {str_txt}")
        print(f"Two trails for best kernels are {best_kernels}")
        print(f"Test arruracy of Two trails are: {scores}")
        print(f"Mean accuracy for two trails: {np.mean(scores): .4f}")


if __name__ == "__main__":
    FILE_PATH = "./digits4000_txt"

    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                        choices=["kNN", "LR", "Per", "SVM"],
                        help='Which model.')
    parser.add_argument('--PCA', type=int, default=0, 
                        choices=[0, 1],
                        help='PCA or Not.') 
    args = parser.parse_args()
    model = args.model
    PCA_flag = False
    str_txt = "Without PCA"
    if args.PCA == 1:
        PCA_flag = True
        str_txt = "With PCA"

    # read data
    data_dict = get_data(FILE_PATH)
    
    if model == "kNN":
        # kNN
        kNN_function(data_dict, PCA_flag, str_txt)
    elif model == "LR":
        # LR
        LR_function(data_dict, PCA_flag, str_txt)
    elif model == "Per":
        # Perceptron
        Perceptron_function(data_dict, PCA_flag, str_txt)
    else:
        # SVM
        SVM_function(data_dict, PCA_flag, str_txt)
