import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


# read in files include local graph properties
polygon_graphs = pd.merge(pd.read_csv('../../_data/result/polygon_basic_metrics.csv'),\
                        pd.read_csv('../../_data/result/polygon_advanced_metrics_labels.csv'), 
                        on = 'Contract') 

ethereum_graphs = pd.merge(pd.read_csv('../../_data/result/ethereum_basic_metrics.csv'),\
                        pd.read_csv('../../_data/result/ethereum_advanced_metrics_labels.csv'), 
                        on = 'Contract')


bnb_graphs =  pd.merge(pd.read_csv('../../_data/result/bnb_basic_metrics.csv'),\
                        pd.read_csv('../../_data/result/bnb_advanced_metrics_labels.csv'), 
                        on = 'Contract')


plt.rcParams.update({'font.size': 20})


graphs = bnb_graphs.append(ethereum_graphs)
graphs = graphs.append(polygon_graphs)


filtered_graph = graphs[['Contract', 'Chain', 'Class']]

class_counts = filtered_graph['Class'].value_counts().reset_index()
class_counts.columns = ['Class', 'Counts']
class_counts['Category'] = class_counts.reset_index().index

result_graph = filtered_graph.merge(class_counts[['Class', 'Category']], on='Class')
result_graph[['Chain', 'Contract', 'Category']].to_csv('labels.csv', index = 0)
top_classes = graphs['Class'].value_counts().head(5).index.tolist()
graphs_filter = graphs.query('Class in @top_classes')


metrics = ['Num_edges', 'Assortativity', 'Reciprocity',
           'Effective_Diameter', 'Clustering_Coefficient']

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))  

axes = axes.flatten()

class_order = graphs_filter['Class'].value_counts().index

for i, metric in enumerate(metrics):
    sns.boxplot(x='Class', y=metric, data=graphs_filter, order=class_order, ax=axes[i])
    axes[i].set_title(f'{metric}', fontsize=20)  
    if metric in ['Num_edges', 'Effective_Diameter']: 
        axes[i].set_yscale('log')
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.93), title='Blockchain',
                    fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.show()


palette = sns.color_palette('Set2') 
chain_colors = {
    'Polygon': palette[0],
    'Ethereum': palette[1],
    'BNB': palette[2]
}


def plot_grouped_boxplot(data_polygon, data_ethereum, data_bnb, ax, 
                         metrics, labels, log = True):
    data_polygon['Chain'] = 'Polygon'
    data_ethereum['Chain'] = 'Ethereum'
    data_bnb['Chain'] = 'BNB'
    
    combined_data = pd.concat([data_polygon[metrics + ['Chain']],
                               data_ethereum[metrics + ['Chain']],
                               data_bnb[metrics + ['Chain']]])
    
    melted_data = combined_data.melt(id_vars='Chain', var_name='Metric', value_name='Value')
    
    sns.boxplot(data=melted_data, x='Metric', y='Value', hue='Chain', ax=ax, palette='Set2')
    
    if log:
        ax.set_yscale('log')
        
    ax.set_xlabel('')  
    ax.set_ylabel('') 
    ax.tick_params(axis='both', labelsize=20)  
    ax.set_xticklabels(labels, rotation=0) 
    ax.get_legend().remove()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Subfigure 1: Number of Nodes and Number of Edges
plot_grouped_boxplot(polygon_graphs, ethereum_graphs, bnb_graphs, axes[0],
                     ['Num_nodes', 'Num_edges'], #'Number of Nodes and Edges', 
                     ['Num nodes', 'Num edges'], True)
axes[0].text(0.5, -0.13, '(a)', ha='center', va='top', transform=axes[0].transAxes, fontsize=20)

# Subfigure 2: Density, Reciprocity, Clustering Coefficient
plot_grouped_boxplot(polygon_graphs, ethereum_graphs, bnb_graphs, axes[1],
                     ['Reciprocity', 'Clustering_Coefficient'],
                     #'Reciprocity and Clustering Coefficient',
                     ['Reciprocity', 'Clustering'], False)
axes[1].text(0.5, -0.13, '(b)', ha='center', va='top', transform=axes[1].transAxes, fontsize=20)

# Subfigure 3: Assortativity
plot_grouped_boxplot(polygon_graphs, ethereum_graphs, bnb_graphs, axes[2],
                     ['Assortativity'],
                     #'Assortativity',
                     ['Assortativity'], False)
axes[2].text(0.5, -0.13, '(c)', ha='center', va='top', transform=axes[2].transAxes, fontsize=20)

# Subfigure 4: Effective Diameter
plot_grouped_boxplot(polygon_graphs, ethereum_graphs, bnb_graphs, axes[3],
                     ['Effective_Diameter'], #'Effective Diameter', 
                     ['Effective Diameter'], True)
axes[3].text(0.5, -0.13, '(d)', ha='center', va='top', transform=axes[3].transAxes, fontsize=20)

handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.89, 0.93), title='Blockchain',
                    fontsize=14, title_fontsize=14)

plt.tight_layout()
# plt.savefig("../figures/cross_chain_properties.eps", format='eps')
plt.show()

