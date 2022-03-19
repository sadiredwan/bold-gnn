import sys
import pandas as pd
import numpy as np
import networkx as nx


def corr_coefs(data):
	np.seterr(divide='ignore', invalid='ignore')
	corr = {}
	for roi in data.columns:
		for proi in data.columns:
			coef = np.corrcoef(data[roi], data[proi])[0, 1]
			if (proi, roi) not in corr:
				corr[(roi, proi)] = coef
	return corr


def build_graph(pearson_matrix):
	G = nx.Graph()
	for key in pearson_matrix:
		if (key[0] != key[1]) and (abs(pearson_matrix[key]) > 0.7):
			G.add_edge(key[0], key[1], weight=pearson_matrix[key])
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

pearson_matrix1 = corr_coefs(df1)
pearson_matrix2 = corr_coefs(df2)

G1 = build_graph(pearson_matrix1)
G2 = build_graph(pearson_matrix2)

laplacian1 = nx.spectrum.laplacian_spectrum(G1)
laplacian2 = nx.spectrum.laplacian_spectrum(G2)

k1 = select_k(laplacian1)
k2 = select_k(laplacian2)

k = min(k1, k2)

similarity = sum((laplacian1[:k] - laplacian2[:k])**2)

print(similarity)
