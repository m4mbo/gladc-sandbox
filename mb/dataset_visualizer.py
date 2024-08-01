import sys
sys.path.append('../')  

import os
from plotnine import ggplot, aes, geom_boxplot, geom_jitter, labs, theme, element_blank, ggsave
from data.load_data import read_graphfile
import pandas as pd

pre_split = False
datadir = "../datasets"
DS = "NCI1"
max_nodes = 0
output_dir = "../graphs"
output_file = os.path.join(output_dir, f"{DS}_boxplot_nodes.png")

if pre_split:
    graphs_train = read_graphfile(datadir, DS + '_training', max_nodes=max_nodes)
    graphs_test = read_graphfile(datadir, DS + '_testing', max_nodes=max_nodes)
    graphs = graphs_train + graphs_test
else:
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)


### Nodes ###

num_nodes = [G.number_of_nodes() for G in graphs]

df = pd.DataFrame({'Number of Nodes': num_nodes, 'Category': ['All']*len(num_nodes)})

plot = (
    ggplot(df, aes(x='Category', y='Number of Nodes')) +
    geom_boxplot(fill='black', alpha=0.2) +  # Make the box transparent
    labs(title=f'Number of Nodes in {DS}', x='', y='Number of Nodes') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank())  # Hide x-axis text and ticks
)

os.makedirs(output_dir, exist_ok=True)

ggsave(plot, filename=output_file, dpi=300, format='png')

### Edges ###

output_file = os.path.join(output_dir, f"{DS}_boxplot_edges.png")

num_edges = [G.number_of_edges() for G in graphs]

df = pd.DataFrame({'Number of Edges': num_edges, 'Category': ['All']*len(num_edges)})

plot = (
    ggplot(df, aes(x='Category', y='Number of Edges')) +
    geom_boxplot(fill='black', alpha=0.2) +  # Make the box transparent
    labs(title=f'Number of Edges in {DS}', x='', y='Number of Edges') +
    theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank())  # Hide x-axis text and ticks
)

os.makedirs(output_dir, exist_ok=True)

ggsave(plot, filename=output_file, dpi=300, format='png')
