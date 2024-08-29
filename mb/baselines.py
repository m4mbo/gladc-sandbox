import sys
sys.path.append('../')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from data.load_data import read_graphfile
import numpy as np
from random import shuffle
import math 


def get_metrics(graphs):
    metrics = []

    for graph in graphs:
        degrees = [d for _, d in graph.degree()]
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()

        if degrees:
            min_deg = min(degrees) 
            max_deg = max(degrees) 
            deg_avg = np.mean(degrees) 
            deg_up_quartile = np.percentile(degrees, 75) 
            deg_down_quartile = np.percentile(degrees, 25) 
            
            if deg_up_quartile != deg_down_quartile:
                iqr = deg_up_quartile - deg_down_quartile
                outlier_threshold_low = deg_down_quartile - 1.5 * iqr
                outlier_threshold_high = deg_up_quartile + 1.5 * iqr
                deg_num_outliers = len([d for d in degrees if d < outlier_threshold_low or d > outlier_threshold_high])
            else:
                deg_num_outliers = 0  # No outliers if quartiles are the same
        else:
            # Default values when no nodes are in the graph
            min_deg = 0
            max_deg = 0
            deg_avg = 0
            deg_up_quartile = 0
            deg_down_quartile = 0
            deg_num_outliers = 0

        # add deg std?
        # add percentile?
        # add median, mode?
        # distance between quartiles?
        # distance between mean and median?
        metrics.append({
            'min_deg': min_deg,
            'max_deg': max_deg,
            'deg_avg': deg_avg,
            'deg_num_outliers': deg_num_outliers,
            'deg_up_quartile': deg_up_quartile,
            'deg_down_quartile': deg_down_quartile,
            'num_edges': num_edges,
            'num_nodes': num_nodes
        })

    return metrics

def getFeaturesAndLabels(graphs_train, graphs_test):

        # Convert the list of dictionaries to a numpy array for model training
    X_train = np.array([[m['min_deg'], m['max_deg'], m['deg_avg'], m['deg_num_outliers'], 
                m['deg_up_quartile'], m['deg_down_quartile'], m['num_edges'], 
                m['num_nodes']] for m in get_metrics(graphs_train)])
    X_test = np.array([[m['min_deg'], m['max_deg'], m['deg_avg'], m['deg_num_outliers'], 
                m['deg_up_quartile'], m['deg_down_quartile'], m['num_edges'], 
                m['num_nodes']] for m in get_metrics(graphs_test)])

    # Labels
    y_train = np.array([G.graph['label'] for G in graphs_train])
    y_test = np.array([G.graph['label'] for G in graphs_test])
    
    return X_train, X_test, y_train, y_test

def IF(X_train, X_test, y_train, y_test):
    if_model = IsolationForest(contamination=0.49, random_state=42, n_estimators=400)
    if_model.fit(X_train)
    y_scores = if_model.decision_function(X_test)
    # Calculate AUC score
    return roc_auc_score(y_test, y_scores)

def RF(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(random_state=42, n_estimators=400)
    rf_model.fit(X_train, y_train)
    y_probs = rf_model.predict_proba(X_test)[:, 1]  
    return roc_auc_score(y_test, y_probs)
   
def DT(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_probs_dt = dt_model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_probs_dt)
  
    
# Load the dataset
datadir = "../datasets"
DS = "NCI1"
max_nodes = 0

if_auc = []
rf_auc = []
dt_auc = []

if DS != "NCI1":
    graphs_train = read_graphfile(datadir, DS + '_training', max_nodes=max_nodes)
    graphs_test = read_graphfile(datadir, DS + '_testing', max_nodes=max_nodes)

    for i in range(5):

        print("Trial {}:".format(i+1))   

        train_num=len(graphs_train)
        all_idx = [idx for idx in range(train_num)]
        shuffle(all_idx)
        num_train=math.ceil(1*train_num)
        train_index = all_idx[:num_train]
        graphs_train = [graphs_train[i] for i in train_index]

        X_train, X_test, y_train, y_test = getFeaturesAndLabels(graphs_train, graphs_test)

        if_auc.append(IF(X_train, X_test, y_train, y_test))
        rf_auc.append(RF(X_train, X_test, y_train, y_test))
        dt_auc.append(DT(X_train, X_test, y_train, y_test))
    
else:

    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)

    graphs_label = np.array([G.graph['label'] for G in graphs])

    kfd=StratifiedKFold(n_splits=5, random_state=42, shuffle = True) # 5 fold

    for k, (train_index,test_index) in enumerate(kfd.split(graphs, graphs_label)):

        graphs_train = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]

        print(f"Fold {k} (train-test split):", len(train_index), len(test_index))

        X_train, X_test, y_train, y_test = getFeaturesAndLabels(graphs_train, graphs_test)

        if_auc.append(IF(X_train, X_test, y_train, y_test))
        rf_auc.append(RF(X_train, X_test, y_train, y_test))
        dt_auc.append(DT(X_train, X_test, y_train, y_test))
        
print(f"IF AUC: {np.mean(if_auc)} +- {np.std(if_auc)}")
print(f"RF AUC: {np.mean(rf_auc)} +- {np.std(rf_auc)}")
print(f"DT AUC: {np.mean(dt_auc)} +- {np.std(dt_auc)}")





