import sys
sys.path.append('../')

import os
from plotnine import ggplot, aes, geom_boxplot, geom_jitter, labs, theme, element_blank, ggsave, geom_density
from data.load_data import read_graphfile
import pandas as pd
import random

pre_split = True
datadir = "../datasets"
DS = "Tox21_HSE"
max_nodes = 0
output_dir = "../graphs"
output_file_nodes = os.path.join(output_dir, f"{DS}_boxplot_nodes.png")
output_file_edges = os.path.join(output_dir, f"{DS}_boxplot_edges.png")
output_file_anomaly_degrees = os.path.join(output_dir, f"{DS}_boxplot_degrees_anomaly.png")
output_file_normal_degrees = os.path.join(output_dir, f"{DS}_boxplot_degrees_normal.png")
output_file_degree_density = os.path.join(output_dir, f"{DS}_degree_density.png")

if pre_split:
    graphs_train = read_graphfile(datadir, DS + '_training', max_nodes=max_nodes)
    graphs_test = read_graphfile(datadir, DS + '_testing', max_nodes=max_nodes)
    graphs = graphs_train + graphs_test
else:
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)


num_nodes = [G.number_of_nodes() for G in graphs]
colors = ['blue' if G.graph["label"] == 0 else 'red' for G in graphs]

df_nodes = pd.DataFrame({'Number of Nodes': num_nodes, 'Category': ['All']*len(num_nodes), 'Color': colors})

plot_nodes = (
    ggplot(df_nodes, aes(x='Category', y='Number of Nodes')) +  # Single category for all
    geom_jitter(aes(color='Color'), width=0.25, height=0, size=1.2) +  # Jitter with color
    geom_boxplot(fill='black', alpha=0.4, outlier_alpha=0) +
    labs(title=f'Number of Nodes in {DS}', x='', y='Number of Nodes') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), legend_position='none')
)

os.makedirs(output_dir, exist_ok=True)
ggsave(plot_nodes, filename=output_file_nodes, dpi=300, format='png')

# Prepare data for edges plot
num_edges = [G.number_of_edges() for G in graphs]

df_edges = pd.DataFrame({'Number of Edges': num_edges, 'Category': ['All']*len(num_edges), 'Color': colors})

# Plot edges with different colors
plot_edges = (
    ggplot(df_edges, aes(x='Category', y='Number of Edges')) +  # Single category for all
    geom_jitter(aes(color='Color'), width=0.25, height=0, size=1.2) +  # Jitter with color
    geom_boxplot(fill='black', alpha=0.4, outlier_alpha=0) +
    labs(title=f'Number of Edges in {DS}', x='', y='Number of Edges') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), legend_position='none')
)

ggsave(plot_edges, filename=output_file_edges, dpi=300, format='png')

anomaly_graphs = [G for G in graphs if G.graph["label"] != 0]
rand_anomaly_graph = random.choice(anomaly_graphs)

node_degrees_anomaly = [degree for _, degree in rand_anomaly_graph.degree()]

df_degrees_anomaly = pd.DataFrame({'Node Degrees': node_degrees_anomaly, 'Category': ['All']*len(node_degrees_anomaly)})

plot_edges = (
    ggplot(df_degrees_anomaly, aes(x='Category', y='Node Degrees')) +  # Single category for all
    geom_jitter(width=0.25, height=0, size=1.2) +  # Jitter with color
    geom_boxplot(fill='black', alpha=0.4, outlier_alpha=0) +
    labs(title=f'Node Degree in Random Anomaly Graph {DS}', x='', y='Node Degrees') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), legend_position='none')
)

ggsave(plot_edges, filename=output_file_anomaly_degrees, dpi=300, format='png')

normal_graphs = [G for G in graphs if G.graph["label"] == 0]
rand_normal_graph = random.choice(normal_graphs)

node_degrees_normal = [degree for _, degree in rand_normal_graph.degree()]

df_degrees_normal = pd.DataFrame({'Node Degrees': node_degrees_normal, 'Category': ['All']*len(node_degrees_normal)})

plot_edges = (
    ggplot(df_degrees_normal, aes(x='Category', y='Node Degrees')) +  # Single category for all
    geom_jitter(width=0.25, height=0, size=1.2) +  # Jitter with color
    geom_boxplot(fill='black', alpha=0.4, outlier_alpha=0) +
    labs(title=f'Node Degrees in Random Normal Graph: {DS}', x='', y='Node Degrees') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), legend_position='none')
)

ggsave(plot_edges, filename=output_file_normal_degrees, dpi=300, format='png')

# Prepare data for average node degree density plot
average_degrees_normal = [sum(dict(G.degree()).values()) / G.number_of_nodes() for G in normal_graphs if G.number_of_nodes() != 0]
average_degrees_anomaly = [sum(dict(G.degree()).values()) / G.number_of_nodes() for G in anomaly_graphs if G.number_of_nodes() != 0]

df_density = pd.DataFrame({
    'Average Node Degree': average_degrees_normal + average_degrees_anomaly,
    'Type': ['Normal']*len(average_degrees_normal) + ['Anomaly']*len(average_degrees_anomaly)
})

# Plot density
plot_density = (
    ggplot(df_density, aes(x='Average Node Degree', color='Type', fill='Type')) +
    geom_density(alpha=0.4) +
    labs(title=f'Density Plot of Average Node Degrees in {DS}', x='Average Node Degree', y='Density') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), legend_position='right')
)

ggsave(plot_density, filename=output_file_degree_density, dpi=300, format='png')
