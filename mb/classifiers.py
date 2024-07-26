import sys
sys.path.append('../')  

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from data.load_data import *
from data.graph_build import GraphBuild

def get_graphs(datadir: str,
               DS: str,
               max_nodes: int,
               pre_split: bool=False,
               random_state: int=None):
    
    if pre_split:
        graphs_train = read_graphfile(datadir, DS+'_training', max_nodes=max_nodes)
        graphs_test = read_graphfile(datadir, DS+'_testing', max_nodes=max_nodes)  
    else:
        graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)  
        graphs_train, graphs_test = train_test_split(graphs, test_size=0.2, random_state=random_state)

    return graphs_train, graphs_test

def extract_features_and_labels(graphs, max_nodes):
    
    if max_nodes == 0:
        max_nodes = max([G.number_of_nodes() for G in graphs])

    dataset = GraphBuild(graphs, features='deg-num', normalize=False, max_num_nodes=max_nodes)
    
    X = np.array([sample['feats'].flatten() for sample in dataset])
    y = np.array([1 if sample['label'] == 0 else 0 for sample in dataset])  # 1 for abnormality, 0 for normal
    return X, y

def dummy_AUC(datadir: str,
              DS: str,
              max_nodes: int,
              pre_split: bool=False,
              random_state: int=None):

    graphs_train, graphs_test = get_graphs(datadir, DS, max_nodes, pre_split, random_state)
    
    X_train, y_train = extract_features_and_labels(graphs_train, max_nodes)
    X_test, y_test = extract_features_and_labels(graphs_test, max_nodes)
    
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    
    y_pred_proba = dummy_clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    return auc_score
    

def naive_AUC():
    return 1


if __name__ == '__main__':
    datadir = "../datasets"
    DS = "Tox21_HSE"
    max_nodes = 0
    auc_score = dummy_AUC(datadir, DS, max_nodes, pre_split=True)
    print("Dummy Classifier AUC:", auc_score)
