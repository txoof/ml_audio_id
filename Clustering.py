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

# +
import numpy as np
import librosa
from pathlib import Path
import pickle
import json
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os
from pydub import AudioSegment
from pydub.playback import play
from IPython.display import Audio, display



def load_audio(file_path, sr=22050):
    """
    Load an audio file using librosa.

    Parameters:
    file_path (str): Path to the audio file.
    sr (int): Sampling rate for loading the audio. Default is 22050 Hz.

    Returns:
    tuple: (audio time series, sampling rate)
    """
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def extract_features(y, sr):
    """
    Extract features from an audio time series.

    Parameters:
    y (numpy.ndarray): Audio time series.
    sr (int): Sampling rate of the audio.

    Returns:
    dict: Extracted features including MFCCs, chroma, spectral contrast, and zero-crossing rate.
    """
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Extract Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    
    return {
        'mfccs': mfccs,
        'chroma': chroma,
        'spectral_contrast': spectral_contrast,
        'zero_crossing_rate': zero_crossing_rate
    }

def process_audio_directory(directory_path, output_file='./extracted_features.pkl'):
    """
    Process all audio files in a directory to extract features and save them to a pickle file.

    Parameters:
    directory_path (str): Path to the directory containing audio files.
    output_file (str): Path to the pickle file where features will be saved.

    Returns:
    list: List of dictionaries containing features for each file.
    """
    directory = Path(directory_path)
    features_list = []
    for file_path in directory.glob('*.mp3'):
        y, sr = load_audio(str(file_path))
        features = extract_features(y, sr)
        features_list.append({'file_name': file_path.name, 'features': features})
    
    # Write features to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(features_list, f)
    
    return features_list

def load_audio_features(pickle_file):
    """
    Load audio features from a pickle file.

    Parameters:
    pickle_file (str): Path to the pickle file containing audio features.

    Returns:
    list: List of dictionaries containing features for each file.
    """
    with open(pickle_file, 'rb') as f:
        features_list = pickle.load(f)
    return features_list

def prepare_features_for_clustering(features_list):
    """
    Prepare features for clustering by converting them into a suitable format.

    Parameters:
    features_list (list): List of dictionaries containing features for each file.

    Returns:
    numpy.ndarray: A 2D array where each row represents concatenated features of an audio file.
    """
    feature_vectors = []
    for item in features_list:
        features = item['features']
        # Flatten and concatenate features
        feature_vector = np.concatenate([
            np.mean(features['mfccs'], axis=1),
            np.mean(features['chroma'], axis=1),
            np.mean(features['spectral_contrast'], axis=1),
            np.mean(features['zero_crossing_rate'], axis=1)
        ])
        feature_vectors.append(feature_vector)
    return np.array(feature_vectors)

def cluster_features_hierarchical(feature_vectors, n_clusters=4):
    """
    Cluster the feature vectors using Agglomerative Clustering and plot a dendrogram.

    Parameters:
    feature_vectors (numpy.ndarray): A 2D array where each row represents features of an audio file.
    n_clusters (int): The number of clusters to form. Default is 4.

    Returns:
    numpy.ndarray: Array of cluster labels for each feature vector.
    """
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(feature_vectors)
    
    # Plot dendrogram
    linkage_matrix = sch.linkage(feature_vectors, method='ward')
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage_matrix)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Audio Files")
    plt.ylabel("Distance")
    plt.show()
    
    return labels

def load_ground_truth_labels(json_file):
    """
    Load ground truth labels from a JSON file.

    Parameters:
    json_file (str): Path to the JSON file containing ground truth labels.

    Returns:
    dict: Dictionary containing file names as keys and labels as values.
    """
    with open(json_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth

def label_clusters(features_list, cluster_labels, ground_truth, n_samples=10, file_path='./path_to_your_audio_files'):
    """
    Label clusters using ground truth data and manual labeling if needed.

    Parameters:
    features_list (list): List of dictionaries containing features for each file.
    cluster_labels (numpy.ndarray): Array of cluster labels for each feature vector.
    ground_truth (dict): Dictionary containing ground truth labels.
    n_samples (int): Number of manual samples to label.
    file_path (str): Path to the directory containing audio files.

    Returns:
    dict: Dictionary containing cluster labels as keys and assigned tags as values.
    """
    cluster_to_label = {}
    labeled_files = 0
    
    for idx, item in enumerate(features_list):
        file_name = item['file_name']
        cluster_id = cluster_labels[idx]
        
        # Assign label using ground truth if available
        if file_name in ground_truth:
            label = ground_truth[file_name]
            cluster_to_label[cluster_id] = label
        elif isinstance(n_samples, int) and labeled_files < n_samples:
            # Manually label remaining samples if needed
            print(f"File: {file_name}, Cluster: {cluster_id}")
            mp3_file = Path(file_path) / file_name
            # Load and play the audio in Jupyter
            display(Audio(str(mp3_file), autoplay=True))
            label = input("Enter label (Music, Dialogue, Both, None): ")
            cluster_to_label[cluster_id] = label
            labeled_files += 1
    
    return cluster_to_label

def create_m3u_playlists(features_list, cluster_labels, cluster_to_label, directory_path):
    """
    Create M3U playlists for audio files based on their assigned labels.

    Parameters:
    features_list (list): List of dictionaries containing features for each file.
    cluster_labels (numpy.ndarray): Array of cluster labels for each feature vector.
    cluster_to_label (dict): Dictionary containing cluster labels and their assigned tags.
    directory_path (str): Path to the directory containing audio files.
    """
    # Create a playlist file for each label
    labels = set(cluster_to_label.values())
    for label in labels:
        playlist_path = Path(directory_path) / f"{label}.m3u"
        with open(playlist_path, 'w') as playlist_file:
            for idx, item in enumerate(features_list):
                file_name = item['file_name']
                cluster_id = cluster_labels[idx]
                if cluster_id in cluster_to_label and cluster_to_label[cluster_id] == label:
                    playlist_file.write(f"{directory_path}/{file_name}\n")



# +
directory_path = './TRAINING_DATA/PREPARED/outro/'
audio_features = process_audio_directory(directory_path)


directory_path = "./TRAINING_DATA/PREPARED/outro/"
output_file = "./audio_features.pkl"
# audio_features = process_audio_directory(directory_path, output_file)
# print(audio_features)


# Load features from pickle file
loaded_features = load_audio_features(output_file)
feature_vectors = prepare_features_for_clustering(loaded_features)
# -

# Perform clustering
cluster_labels = cluster_features_hierarchical(feature_vectors)

# +
# load ground truth, manually tagged files
ground_truth = load_ground_truth_labels('track_classifications.json')

# cluster_to_label = label_clusters(loaded_features, cluster_labels, ground_truth, directory_path)
cluster_to_label = label_clusters(features_list=loaded_features, 
                                  cluster_labels=cluster_labels, 
                                  ground_truth=ground_truth,
                                  n_samples=25,
                                  file_path=directory_path)
# -

# save current clusters to m3u files for checking
create_m3u_playlists(loaded_features, cluster_labels, cluster_to_label, './')
