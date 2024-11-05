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

# # Hierarchical Cluster Guided Labeling (HCGL)
#
# This method is based on the paper [*Efficient Label Collection for Unlabeled Image Datasets*](./Wigness_Efficient_Label_Collection_2015_CVPR_paper.pdf)

from pathlib import Path
import audio_features
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import fcluster
import random
import IPython.display as ipd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json


def read_classifications(output_file):
    """
    Read the specified file and return the count of each category
    """
    output_file = Path(output_file)
    
    classified_tracks = set()
    classifications = {'M': 0, 'D': 0, 'B': 0}
    if output_file.exists():
        classified_df = pd.read_csv(output_file)
        classified_tracks.update(classified_df['Filename'].tolist())
        # Update the classification counts with existing data
        existing_classifications = classified_df['Classification'].value_counts().to_dict()
        for key in ['M', 'D', 'B']:
            if key in existing_classifications:
                classifications[key] += existing_classifications[key]
                
    total_classifications = sum(classifications.values())

    if total_classifications > 0:
        dialog_pct = classifications["D"] / total_classifications
        music_both_pct = (classifications["M"] + classifications["B"]) / total_classifications    
    return classifications, {'Music & Both': music_both_pct, 'Dialogue': dialog_pct}


import random
import IPython.display as ipd
import time
def play_and_classify_m3u(m3u_file, output_file=None, num_tracks=20):
    """
    Play a specified or random number of items from an M3U file, ask user to classify as M, D, B, 
    and record the values in a file, displaying the ratio of M, D, B.
    Press 'Q' to quit at any time.
    
    Parameters:
    m3u_file (str or Path): Path to the M3U file.
    output_file (str or Path): Path to the output file where classifications are saved.
    num_tracks (int, optional): Number of tracks to play. If None, a random number of tracks will be played.
    """
    m3u_file = Path(m3u_file)

    if not output_file:
        output_file = m3u_file.parent / (m3u_file.stem + '_classification.csv')
    else:
        output_file = Path(output_file)
    
    # Read M3U file
    with m3u_file.open('r') as f:
        lines = f.readlines()
    
    # Filter lines to get only filenames (skip #EXTM3U and other comments)
    tracks = [line.strip() for line in lines if not line.startswith('#')]
    
    # Read existing classifications if the output file exists
    classified_tracks = set()
    classifications = {'M': 0, 'D': 0, 'B': 0}
    if output_file.exists():
        # classified_df = pd.read_csv(output_file)
        # classified_tracks.update(classified_df['Filename'].tolist())
        # # Update the classification counts with existing data
        # existing_classifications = classified_df['Classification'].value_counts().to_dict()
        # for key in ['M', 'D', 'B']:
        #     if key in existing_classifications:
        #         classifications[key] += existing_classifications[key]
        classifications, totals = read_classifications(output_file)
    else:
        with output_file.open('w') as out_file:
            out_file.write('Filename,Classification\n')
    
    # Filter out tracks that have already been classified
    tracks_to_play = [track for track in tracks if track not in classified_tracks]
    
    # Randomly shuffle the list of tracks to play
    random.shuffle(tracks_to_play)
    
    # Determine the number of tracks to play
    num_tracks = min(num_tracks, len(tracks_to_play))
    
    # Open output file for appending classifications
    with output_file.open('a') as out_file:
        # Play each track and ask for classification
        for i in range(num_tracks):
            track = tracks_to_play[i]
            print(f"Playing track {i + 1} of {num_tracks}: {track}")
            
            # Automatically play the audio file
            audio = ipd.Audio(track, autoplay=True)
            display_handle = ipd.display(audio, display_id=True)
            
            # Get user classification (M, D, B, or Q to quit)
            classification = None
            while classification not in ['M', 'D', 'B', 'Q']:
                classification = input("Classify this track as M (Music), D (Dialogue), B (Both), or Q (Quit): ").upper()
                if classification == 'Q':
                    print("Quitting...")
                    return
            
            # Record classification if not quitting
            if classification != 'Q':
                classifications[classification] += 1
                out_file.write(f"{track},{classification}\n")
            
            # Pause between tracks
            time.sleep(1)
            display_handle.update(ipd.HTML(''))
    
    # Calculate and display ratio of classifications
    total_classifications = sum(classifications.values())
    print("\nClassification Ratios:")
    for key, value in classifications.items():
        ratio = value / total_classifications if total_classifications > 0 else 0
        print(f"{key}: {ratio:.2f}")
    dialog_pct = classifications["D"] / total_classifications
    music_both_pct = (classifications["M"] + classifications["B"]) / total_classifications
    print(f"Music & Both: {music_both_pct * 100}; Dialogue: {dialog_pct * 100}")
    return classifications


# ## Load extracted features
#
# Features such as chroma, zero-mean-crossing, RMS energy, etc. have been extracted and stored in a CSV file. 

# +
feature_file = Path('241103_1441_features.csv')
training_data = Path('./TRAINING_DATA/PREPARED/outro/')

features, filenames = audio_features.read_features_from_file(feature_file)
# -

# ## Perform Hiearchical Clustering
#
# Use scipy linkages method to agglomerate the features. Various strategies are tested here. The goal is to find the best vertical cutoff that shows groupings that are easy to distinguish, have low vertical distance within the clusters and high vertical distance between other clusters. It appears that the **ward**, **median** and **average** mtehods are the most promising for distinguishing between tracks that contain *some* music versus clusters that contain only dialogue. 

for i in ['single', 'complete', 'average', 'ward', 'weighted', 'centroid', 'median']:
    linkage_matrix = sch.linkage(features, method=i)
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage_matrix, no_labels=True)
    plt.title(f'Hierarchical Clustering Dendrogram: {i}')
    plt.xlabel('Sample Index (Cluster Size)')
    plt.ylabel('Distance')
    plt.show()

# ### Test Methods



# +
# Extract cluster assignments by cutting the dendrogram
# Cut the dendrogram at a specific distance to get 3 clusters
## MEDIAN 
linkage_matrix = sch.linkage(features, method='median')
distance_threshold = 5000  # Adjust this value as needed to separate clusters
cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

# Create a DataFrame to analyze clusters
cluster_data = pd.DataFrame({'Filename': filenames, 'Cluster': cluster_labels})

# Investigate the orange and green clusters (assuming clusters are labeled 1, 2, 3)
orange_cluster = cluster_data[cluster_data['Cluster'] == 1]
green_cluster = cluster_data[cluster_data['Cluster'] == 2]


write_m3u(orange_cluster, 'orange_median.m3u')
write_m3u(green_cluster, 'green_median.m3u')

play_and_classify_m3u('orange_median.m3u', num_tracks=1)

play_and_classify_m3u('green_median.m3u', num_tracks=1)

# +
# Extract cluster assignments by cutting the dendrogram
# Cut the dendrogram at a specific distance to get 3 clusters
## WARD 
linkage_matrix = sch.linkage(features, method='ward')
distance_threshold = 20000  # Adjust this value as needed to separate clusters
cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

# Create a DataFrame to analyze clusters
cluster_data = pd.DataFrame({'Filename': filenames, 'Cluster': cluster_labels})

# Investigate the orange and green clusters (assuming clusters are labeled 1, 2, 3)
orange_cluster = cluster_data[cluster_data['Cluster'] == 1]
green_cluster = cluster_data[cluster_data['Cluster'] == 2]


write_m3u(orange_cluster, 'orange_ward.m3u')
write_m3u(green_cluster, 'green_ward.m3u')

play_and_classify_m3u('orange_ward_.m3u', num_tracks=20)

play_and_classify_m3u('green_ward.m3u', num_tracks=20)

# +
# Extract cluster assignments by cutting the dendrogram
# Cut the dendrogram at a specific distance to get 3 clusters
## WARD 
linkage_matrix = sch.linkage(features, method='complete')
distance_threshold = 10000  # Adjust this value as needed to separate clusters
cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

# Create a DataFrame to analyze clusters
cluster_data = pd.DataFrame({'Filename': filenames, 'Cluster': cluster_labels})

# Investigate the orange and green clusters (assuming clusters are labeled 1, 2, 3)
orange_cluster = cluster_data[cluster_data['Cluster'] == 1]
green_cluster = cluster_data[cluster_data['Cluster'] == 2]


write_m3u(orange_cluster, 'orange_complete.m3u')
write_m3u(green_cluster, 'green_complete.m3u')

play_and_classify_m3u('orange_complete.m3u', num_tracks=20)

play_and_classify_m3u('green_complete.m3u', num_tracks=20)
# -

# ## Linkage Method Results

# ### Median
#
# **Green Cluster**: 
#
# - Music & Both: 33%
# - Dialogue: 67%
#
# **Orange Cluster**:
#
# - Music & Both: 67%
# - Dialogue: 32%

print(read_classifications('./green_median_classification.csv'))
print(read_classifications('./orange_median_classification.csv'))

# ### Complete
#
# **Green Cluster**:
#
# - Music & Both: 50%
# - Dialogue: 50%
#
# **Oragne Cluster**:
#
# - Music & Both: 95%
# - Dialogue: 5%

print(read_classifications('./green_complete_classification.csv'))
print(read_classifications('./orange_complete_classification.csv'))

# ### Ward
#
# **Green Cluster**:
#
# - Music & Both: 37%
# - Dialogue: 63%
#
# **Oragne Cluster**:
#
# - Music & Both: 100%
# - Dialogue: 0%

print(read_classifications('./green_ward_classification.csv'))
print(read_classifications('./orange_ward_classification.csv'))
