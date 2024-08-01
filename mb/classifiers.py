import sys
sys.path.append('../')  

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from data.load_data import *
from data.graph_build import GraphBuild
from random import shuffle
import math

class Classifier:
    def __init__(self, datadir, max_nodes, DS, pre_split: bool):
        self.pre_split = pre_split
        self.max_nodes = max_nodes

        if pre_split:
            self.graphs_train = read_graphfile(datadir, DS + '_training', max_nodes=max_nodes)
            self.graphs_test = read_graphfile(datadir, DS + '_testing', max_nodes=max_nodes)
        else:
            self.graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)

    def classify(self):
        pass

    def extract_features_and_labels(self, graphs, max_nodes):
        if max_nodes == 0:
            max_nodes = max([G.number_of_nodes() for G in graphs])

        dataset = GraphBuild(graphs, features='deg-num', normalize=False, max_num_nodes=max_nodes)
        
        X = np.array([sample['feats'].flatten() for sample in dataset])
        y = np.array([0 if sample['label'] == 1 else 1 for sample in dataset])  # 1 for abnormality, 0 for normal
        return X, y
    
class NaiveClassifier(Classifier):
    def __init__(self, datadir, max_nodes, DS, pre_split: bool):
        super().__init__(datadir, max_nodes, DS, pre_split)

    def get_metrics(self, graphs):
        min_deg = float('inf')
        max_deg = float('-inf')

        min_nodes = float('inf')
        max_nodes = float('-inf')

        min_edges = float('inf')
        max_edges = float('-inf')

        for graph in graphs:
            degrees = [d for _, d in graph.degree()]
            num_edges = graph.number_of_edges()
            num_nodes = graph.number_of_nodes()

            if degrees:
                min_deg = min(min_deg, min(degrees))
                max_deg = max(max_deg, max(degrees))
            
            min_edges = min(min_edges, num_edges)
            max_edges = max(max_edges, num_edges)
            
            min_nodes = min(min_nodes, num_nodes)
            max_nodes = max(max_nodes, num_nodes)

        deg = {'min_deg': min_deg, 'max_deg': max_deg}
        edges = {'min_edges': min_edges, 'max_edges': max_edges}
        nodes = {'min_nodes': min_nodes, 'max_nodes': max_nodes}

        return deg, edges, nodes
    
    def classify_pre(self, result_auc):
        for i in range(5):
            train_num = len(self.graphs_train)
            all_idx = list(range(train_num))
            shuffle(all_idx)
            num_train = math.ceil(1 * train_num)
            train_index = all_idx[:num_train]
            self.graphs_train = [self.graphs_train[i] for i in train_index]
            normal = [graph for graph in self.graphs_train if graph.graph['label'] == 0]
            deg, edges, nodes = self.get_metrics(normal)

            y_test = []
            y_pred = []
            for graph in self.graphs_test:
                y_test.append(graph.graph['label'])
                is_abnormal = (graph.number_of_nodes() < nodes['min_nodes'] or 
                               graph.number_of_nodes() > nodes['max_nodes'] or
                               graph.number_of_edges() < edges['min_edges'] or 
                               graph.number_of_edges() > edges['max_edges'] or
                               min([d for _, d in graph.degree()]) < deg['min_deg'] or 
                               max([d for _, d in graph.degree()]) > deg['max_deg'])
                y_pred.append(1 if is_abnormal else 0)

            auc_score = roc_auc_score(y_test, y_pred)
            result_auc.append(auc_score)

    def classify(self):

        result_auc = []
        if self.pre_split:
            self.classify_pre(result_auc)
        else:
            graphs_label = [graph.graph['label'] for graph in self.graphs]
            kfd = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # 5 fold
            for train_index, test_index in kfd.split(self.graphs, graphs_label):
                self.graphs_train = [self.graphs[i] for i in train_index]
                self.graphs_test = [self.graphs[i] for i in test_index]
                normal = [graph for graph in self.graphs_train if graph.graph['label'] == 0]
                deg, edges, nodes = self.get_metrics(normal)

                y_test = []
                y_pred = []
                for graph in self.graphs_test:
                    y_test.append(graph.graph['label'])
                    is_abnormal = (graph.number_of_nodes() < nodes['min_nodes'] or 
                                   graph.number_of_nodes() > nodes['max_nodes'] or
                                   graph.number_of_edges() < edges['min_edges'] or 
                                   graph.number_of_edges() > edges['max_edges'] or
                                   min([d for _, d in graph.degree()]) < deg['min_deg'] or 
                                   max([d for _, d in graph.degree()]) > deg['max_deg'])
                    y_pred.append(1 if is_abnormal else 0)

                auc_score = roc_auc_score(y_test, y_pred)
                result_auc.append(auc_score)

        print('AUROC:', result_auc, 'Average:', np.mean(result_auc), 'STD:', np.std(result_auc))

class myDummyClassifier(Classifier):
    def __init__(self, datadir, max_nodes, DS, pre_split: bool):
        super().__init__(datadir, max_nodes, DS, pre_split)
    
    def classify_pre(self, result_auc):
        for i in range(5):
            train_num = len(self.graphs_train)
            all_idx = list(range(train_num))
            shuffle(all_idx)
            num_train = math.ceil(1 * train_num)
            train_index = all_idx[:num_train]
            self.graphs_train = [self.graphs_train[i] for i in train_index]

            X_train, y_train = self.extract_features_and_labels(self.graphs_train, self.max_nodes)
            X_test, y_test = self.extract_features_and_labels(self.graphs_test, self.max_nodes)

            dummy_clf = DummyClassifier(strategy="stratified")
            dummy_clf.fit(X_train, y_train)
            y_pred_proba = dummy_clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
            auc_score = roc_auc_score(y_test, y_pred_proba)
            result_auc.append(auc_score)
        
    def classify(self):
        result_auc = []
        if self.pre_split:
            self.classify_pre(result_auc)
        else:
            graphs_label = [graph.graph['label'] for graph in self.graphs]
            kfd = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # 5 fold
            for train_index, test_index in kfd.split(self.graphs, graphs_label):
                self.graphs_train = [self.graphs[i] for i in train_index]
                self.graphs_test = [self.graphs[i] for i in test_index]

                X_train, y_train = self.extract_features_and_labels(self.graphs_train, self.max_nodes)
                X_test, y_test = self.extract_features_and_labels(self.graphs_test, self.max_nodes)
                dummy_clf = DummyClassifier(strategy="most_frequent")
                dummy_clf.fit(X_train, y_train)
                y_pred_proba = dummy_clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
                auc_score = roc_auc_score(y_test, y_pred_proba)
                result_auc.append(auc_score)
        
        print('auroc{}, average: {}, std: {}'.format(result_auc, np.mean(result_auc), np.std(result_auc)))


if __name__ == '__main__':
    datadir = "../datasets"
    DS = "Tox21_p53"
    max_nodes = 0

    dummy = myDummyClassifier(datadir, max_nodes, DS, True)
    dummy.classify()
    
    
