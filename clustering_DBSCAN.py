# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ml_audio_id-venv-9ab27db4d3
#     language: python
#     name: ml_audio_id-venv-9ab27db4d3
# ---

# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

from audio_features import *

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


feature_file = '241103_1441_features.csv'
training_data = './TRAINING_DATA/PREPARED/outro/'

# Step 1: extract features or load features
# features, filenames = extract_features_from_directory(training_data)
# write_features_to_file(features, filenames, feature_file)
features, filenames = read_features_from_file(feature_file)

# Step 2: Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# +
# Apply PCA to visualize explained variance
pca = PCA()
features_pca = pca.fit_transform(features_normalized)

# Plot cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance Curve')
plt.grid()
plt.show()

# +
pca = PCA(n_components=20)  # Experiment with fewer components
features_reduced = pca.fit_transform(features_normalized)

# Apply DBSCAN on the reduced feature set
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(features_reduced)

# +
best_eps = None
best_min_samples = None
best_num_clusters = 0

for eps in np.linspace(0.01, 3, num=30):
    for min_samples in range(3, 10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_reduced)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)

        if num_clusters > best_num_clusters and num_noise < len(labels) * 0.5:
            best_eps = eps
            best_min_samples = min_samples
            best_num_clusters = num_clusters

print(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best number of clusters: {best_num_clusters}')

# -

float_range = np.linspace(0.01, 0.5, num=20)


# Step 3: Apply DBSCAN
# for i in float_range:
for i in float_range:
    for j in range(2, 5):
        print(f'eps={i}; min_samples={j}')
        dbscan = DBSCAN(eps=i, min_samples=j)  # Adjust `eps` and `min_samples` as needed
        # labels = dbscan.fit_predict(features_normalized)
        labels = dbscan.fit_predict(features_reduced)
        
        # Step 4: Visualize the clustering
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        
        plt.figure(figsize=(10, 6))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = features_normalized[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgewidth=0, markersize=6)
        
        plt.title('DBSCAN Clustering')
        plt.show()


