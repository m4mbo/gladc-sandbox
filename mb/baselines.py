import sys
sys.path.append('../')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from data.load_data import read_graphfile
import numpy as np

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

# Load the dataset
datadir = "../datasets"
DS = "NCI1"
max_nodes = 0

if DS != "NCI1":
    graphs_train = read_graphfile(datadir, DS + '_training', max_nodes=max_nodes)
    graphs_test = read_graphfile(datadir, DS + '_testing', max_nodes=max_nodes)
else:
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    graphs_train, graphs_test = train_test_split(graphs, test_size=0.2, random_state=42)

graphs_train_ = [G for G in graphs_train if G.graph['label'] == 0]

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

if_model = IsolationForest(contamination=0.1, random_state=42)
if_model.fit(X_train)

# Predict anomalies on the test set
y_pred = if_model.predict(X_test)
y_pred_mapped = np.where(y_pred == -1, 1, 0)

y_scores = if_model.decision_function(X_test)

# Calculate AUC score
auc_score_if = roc_auc_score(y_test, y_scores)
print(f"IF AUC Score: {auc_score_if}")

# Feature importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

n_classes = len(rf_model.classes_)
if n_classes > 1:
    # Get probability estimates for the positive class
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    # Calculate AUC score for Random Forest
    auc_score_rf = roc_auc_score(y_test, y_probs)
    print(f"RF AUC Score: {auc_score_rf}")
else:
    print("AUC cannot be computed as only one class is present in y_test.")


importances = rf_model.feature_importances_
feature_names = ['min_deg', 'max_deg', 'deg_avg', 'deg_num_outliers', 
                 'deg_up_quartile', 'deg_down_quartile', 'num_edges', 
                 'num_nodes']

sorted_idx = np.argsort(importances)
plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.show()


