import sys
import numpy as np
import pandas as pd
import nitime.analysis as nta
import nitime.timeseries as ts
import networkx as nx


def coherence(data, f_ub = 0.2, f_lb=0.1):
	time_series = ts.TimeSeries(data.to_numpy().T, sampling_interval=2)
	C = nta.CoherenceAnalyzer(time_series)
	freq_idx_C = np.where((C.frequencies > f_lb) * (C.frequencies < f_ub))[0]
	coh_matrix = np.mean(C.coherence[:, :, freq_idx_C], -1)
	return coh_matrix


def build_graph(coh_matrix):
	G = nx.from_numpy_matrix(coh_matrix, create_using=nx.Graph)
	G.remove_edges_from(nx.selfloop_edges(G))
	sparse = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.3]
	G.remove_edges_from(sparse)
	return G


def select_k(spectrum, threshold=0.9):
	running_total = 0.0
	total = sum(spectrum)
	if total == 0.0:
		return len(spectrum)
	for i in range(len(spectrum)):
		running_total += spectrum[i]
		if running_total / total >= threshold:
			return i + 1
	return len(spectrum)


df1 = pd.read_csv('data/'+'sub-'+sys.argv[1]+'.csv')
df2 = pd.read_csv('data/'+'sub-'+sys.argv[2]+'.csv')

coh_matrix1 = coherence(df1)
coh_matrix2 = coherence(df2)

G1 = build_graph(coh_matrix1)
G2 = build_graph(coh_matrix2)

laplacian1 = nx.spectrum.laplacian_spectrum(G1)
laplacian2 = nx.spectrum.laplacian_spectrum(G2)

k1 = select_k(laplacian1)
k2 = select_k(laplacian2)

k = min(k1, k2)

similarity = sum((laplacian1[:k] - laplacian2[:k])**2)

print(similarity)
