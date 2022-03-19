import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import nitime.analysis as nta
import nitime.timeseries as ts
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from sklearn import model_selection
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


graphs, graph_labels = [], []
lblmap = {
	'1': 'control',
	'5': 'schz',
	'6': 'bipolar',
	'7': 'adhd'
}

print('reading data')

for fname in tqdm(os.listdir('data')):
	df = pd.read_csv('data/'+fname)
	roi_names = df.columns
	time_series = ts.TimeSeries(df.to_numpy().T, sampling_interval=2)

	f_ub = 0.2
	f_lb = 0.1

	C = nta.CoherenceAnalyzer(time_series)

	freq_idx_C = np.where((C.frequencies > f_lb) * (C.frequencies < f_ub))[0]

	coh = np.mean(C.coherence[:, :, freq_idx_C], -1)

	source, target, weight = [], [], []
	for row in range(48):
		for col in range(row+1):
			if coh[row][col] > 0.3 and df.columns[row] != df.columns[col]:
				source.append(df.columns[row])
				target.append(df.columns[col])
				weight.append(coh[row][col])

	nodes = pd.DataFrame(
		{'default': [i for i in range(48)]},
		index=df.columns
	)

	edges = pd.DataFrame(
		{
			'source': source,
			'target': target,
			'weight': weight
		}
	)

	graph = StellarGraph(nodes, edges)

	graphs.append(graph)

	graph_labels.append(lblmap[fname[4]])


graph_labels = pd.get_dummies(graph_labels)

generator = PaddedGraphGenerator(graphs)


k = 4

layer_sizes = [32, 32, 32, 4]

dgcnn_model = DeepGraphCNN(
	layer_sizes=layer_sizes,
	activations=['tanh', 'tanh', 'tanh', 'tanh'],
	k=k,
	bias=False,
	generator=generator,
)

x_in, x_out = dgcnn_model.in_out_tensors()
x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)
x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
x_out = Flatten()(x_out)
x_out = Dense(units=128, activation='relu')(x_out)
x_out = Dropout(rate=0.5)(x_out)
predictions = Dense(units=4, activation='softmax')(x_out)

model = Model(inputs=x_in, outputs=predictions)

model.compile(
	optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['acc'],
)

train_graphs, test_graphs = model_selection.train_test_split(
	graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
	list(train_graphs.index-1),
	targets=train_graphs.values,
	batch_size=2,
	symmetric_normalization=False,
)

test_gen = gen.flow(
	list(test_graphs.index-1),
	targets=test_graphs.values,
	batch_size=2,
	symmetric_normalization=False,
)

history = model.fit(
	train_gen, epochs=100, verbose=1, validation_data=test_gen, shuffle=True,
)
