import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def corr_coefs(data):
	np.seterr(divide='ignore', invalid='ignore')
	corr = {}
	for roi in data.columns:
		for proi in data.columns:
			coef = abs(np.corrcoef(data[roi], data[proi])[0, 1])
			if (proi, roi) not in corr:
				corr[(roi, proi)] = coef
	return corr


def build_graph(pearson_matrix):
	G = nx.Graph()
	for key in pearson_matrix:
		if key[0] != key[1]:
			G.add_edge(key[0], key[1], weight=pearson_matrix[key])
	return G


df = pd.read_csv('data/'+'sub-'+sys.argv[1]+'.csv')

pearson_matrix = corr_coefs(df)

G = build_graph(pearson_matrix)

mapping = {}

for i, column in enumerate(df.columns):
	mapping[column] = i+1

print(mapping)

G = nx.relabel_nodes(G, mapping)

pos = nx.spring_layout(G, seed=10)

high = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.7]

low = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.7]

nx.draw_networkx_edges(G, pos, edgelist=high, edge_color='r')

nx.draw_networkx_edges(G, pos, edgelist=low, alpha=0.3, edge_color='y', style='dashed')

nx.draw_networkx_nodes(G, pos, node_size=200, node_color='w')

nx.draw_networkx_labels(G, pos, font_size=10, font_color='k')

plt.show()
