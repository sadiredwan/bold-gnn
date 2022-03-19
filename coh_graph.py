import sys
import numpy as np
import pandas as pd
import nitime.analysis as nta
import nitime.timeseries as ts
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv('data/'+'sub-'+sys.argv[1]+'.csv')
roi_names = df.columns
time_series = ts.TimeSeries(df.to_numpy().T, sampling_interval=2)

f_ub = 0.2
f_lb = 0.1

C = nta.CoherenceAnalyzer(time_series)

freq_idx_C = np.where((C.frequencies > f_lb) * (C.frequencies < f_ub))[0]

coh = np.mean(C.coherence[:, :, freq_idx_C], -1)

G = nx.from_numpy_matrix(coh, create_using=nx.Graph)

G.remove_edges_from(nx.selfloop_edges(G))

pos = nx.spring_layout(G, seed=10)

high = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]

low = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

nx.draw_networkx_edges(G, pos, edgelist=high, edge_color='r')

nx.draw_networkx_edges(G, pos, edgelist=low, alpha=0.3, edge_color='y', style='dashed')

nx.draw_networkx_nodes(G, pos, node_size=200, node_color='w')

nx.draw_networkx_labels(G, pos, font_size=10, font_color='k')

plt.show()
