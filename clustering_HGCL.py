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


def json_load(labeled_data_file, parent=None):
    """
    Load a JSON file containing label data in the form:
        filename: Music | Dialogue | Both | None
    Prepend parent as a path if provided to the keys in the dictionary

    Parameters:
    labeled_data_file (str or Path): file containing json data
    parent(str or Path): parent path to prepend to the file names
    """
    with open(labeled_data_file, 'r') as f:
        labeled_data = json.load(f)

    if parent:
        return {str(Path(parent) / k): v for k, v in labeled_data.items()}
    else:
        return {str(Path(k)): v for k, v in labeled_data.items()}


def improve_clustering_with_labeled_data(cluster_data, labeled_data_file, label_parent=None):
    """
    Improve clustering accuracy using a labeled dataset.
    
    Parameters:
    cluster_data (pd.DataFrame): DataFrame containing filenames and cluster labels.
    labeled_data_file (str or Path): Path to the labeled dataset JSON file.
    
    Returns:
    pd.DataFrame: Updated cluster data with improved labels.
    """
    # # Load labeled data from JSON file
    labeled_data = json_load(labeled_data_file, label_parent)
    
    # Extract features for labeled data
    labeled_features = []
    labeled_labels = []
    for filename, label in labeled_data.items():
        if filename in cluster_data['Filename'].values:
            index = cluster_data[cluster_data['Filename'] == filename].index[0]
            labeled_features.append(features[index])
            labeled_labels.append(label)
    
    labeled_features = np.array(labeled_features)
    
    # Debugging: Print the number of labeled features found
    print(f"Number of labeled features found: {len(labeled_features)}")
    
    # Check if labeled_features is empty
    if len(labeled_features) == 0:
        raise ValueError("No matching labeled features found. Ensure that the filenames in the labeled data match those in the cluster data.")
    
    # Split labeled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(labeled_features, labeled_labels, test_size=0.2, random_state=42)
    
    # Train a RandomForest classifier on the labeled data
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # Evaluate classifier performance on the test set
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Use the trained classifier to predict labels for the cluster data
    cluster_features = np.array([features[idx] for idx in cluster_data.index])
    predicted_labels = classifier.predict(cluster_features)
    prediction_probs = classifier.predict_proba(cluster_features)

    
    # Add predicted labels to the cluster data
    cluster_data['Improved_Label'] = predicted_labels
    cluster_data['Prediction_Probabilities'] = list(prediction_probs)
        
    return cluster_data, classifier


def manual_classify_uncertain_tracks(cluster_data, classifier, threshold=0.6):
    """
    Manually classify uncertain tracks based on prediction probabilities.
    
    Parameters:
    cluster_data (pd.DataFrame): DataFrame containing filenames, predicted labels, and prediction probabilities.
    classifier (RandomForestClassifier): Trained classifier.
    threshold (float): Probability threshold below which tracks are considered uncertain.
    
    Returns:
    pd.DataFrame: Updated cluster data with manual classifications for uncertain tracks.
    """
    # Identify uncertain tracks
    uncertain_indices = []
    for idx, probs in enumerate(cluster_data['Prediction_Probabilities']):
        max_prob = max(probs)
        if max_prob < threshold:
            uncertain_indices.append(idx)
    
    # Manually classify uncertain tracks
    print(f"Manually reclassifying {len(uncertain_indices)} tracks with uncertainty below {threshold}")

    counter = 1
    
    for idx in uncertain_indices:
        print(f"Track {counter} of {len(uncertain_indices)}")
        counter += 1
        track = cluster_data.iloc[idx]['Filename']
        print(f"Playing uncertain track: {track}")
        
        # Automatically play the audio file
        audio = ipd.Audio(track, autoplay=True)
        display_handle = ipd.display(audio, display_id=True)
        
        # Get user classification (M, D, B, or Q to quit)
        user_input = None
        classification_dict = {
            'M': 'Music',
            'B': 'Both',
            'D': 'Dialogue',
            'Q': 'Quit'
        }
        while user_input not in classification_dict.keys():
            user_input = input("Classify this track as M (Music), D (Dialogue), B (Both), or Q (Quit): ").upper()
            if user_input == 'Q':
                print("Quitting...")
                return cluster_data
        
        # Update the cluster data with the manual classification
        cluster_data.at[idx, 'Improved_Label'] = classification_dict[user_input]
        
        # Pause between tracks
        time.sleep(1)
        display_handle.update(ipd.HTML(''))
    
    return cluster_data


# +
feature_file = Path('241103_1441_features.csv')
training_data = Path('./TRAINING_DATA/PREPARED/outro/')
labeled_json = Path('./track_classifications_241103.json')

features, filenames = audio_features.read_features_from_file(feature_file)
# -

# ## Ward Linkage Dendrogram
#
# Use the "ward" method for determining the cutoff of the feature set. The three clusters appear to diverge around 20,000; a value of 20,100 will be used to split the clusters.

# +
# Perform hierarchical clustering using 'ward' method
linkage_matrix = sch.linkage(features, method='ward')

# Plot dendrogram to visualize clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linkage_matrix, no_labels=True)
plt.title('Hierarchical Clustering Dendrogram: ward')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()
# -

# ## Extract Clusters
#
# Extract the clusters from the whole and generate two groupings: orange and green. Write these out to a playlist that can be manually examined.

# +
# Extract cluster assignments by cutting the dendrogram
# Cut the dendrogram at a specific distance to get 2 clusters
distance_threshold = 20100 # Adjust this value as needed to separate clusters
cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

# Create a DataFrame to analyze clusters
cluster_data = pd.DataFrame({'Filename': filenames, 'Cluster': cluster_labels})

# Investigate the clusters
cluster_1 = cluster_data[cluster_data['Cluster'] == 1]
cluster_2 = cluster_data[cluster_data['Cluster'] == 2]

# Display information about the clusters
print("Orange:")
print(cluster_1)
print("\nGreen 2:")
print(cluster_2)

# Example usage: write M3U files for clusters
audio_features.write_m3u(cluster_1, 'ward_orange_001.m3u')
audio_features.write_m3u(cluster_2, 'ward_green_001.m3u')
# -

# ## Refine Clusters
#
# Use the `improve_clustering_with_labeled_data` function to apply manually labeled items as a ground truth. This should theoretically improve the accuracy.

refined_cluster_data, classifier = improve_clustering_with_labeled_data(cluster_data, 
                                                            labeled_json, training_data)


# ## Manually Classify the Most Uncertain
#
# Manually listten to and classify the most uncertain tracks. A threshold of 60% uncertainty was chosen.

# Refine the clusters using manual classification of uncertain tracks
refined_cluster_data = manual_classify_uncertain_tracks(cluster_data, classifier, threshold=0.6)

# ### Write the refined clusters to m3u 
#
# Write out m3u playlists that can be manually played and classified once again to verify accuracy. These will become the training sets for future training of models.

# +
# save the new label data to a CSV
# refined_cluster_data.to_csv('refined_labels_241105.csv', index=False)

# Load refined cluster data from CSV
refined_csv_path = 'refined_labels_241105.csv'
refined_cluster_data = pd.read_csv(refined_csv_path)

# Write M3U files for newly refined clusters
for label in refined_cluster_data['Improved_Label'].unique():
    subset = refined_cluster_data[refined_cluster_data['Improved_Label'] == label]
    audio_features.write_m3u(subset, f'00_refined_{label.lower()}_playlist.m3u')

# Extract features and improved labels from the refined cluster data
refined_features = []
refined_labels = []

for idx, row in refined_cluster_data.iterrows():
    filename = row['Filename']
    if filename in filenames:
        index = filenames.index(filename)
        refined_features.append(features[index])
        refined_labels.append(row['Improved_Label'])

# Convert to NumPy array for training
refined_features = np.array(refined_features)
# -

refined_cluster_data['Improved_Label'].value_counts()

# ## Verify Refined Clusters
#
# The m3u files containing the three clusters can be sampled and reviewed. The function below plays a random selection from an m3u file and records classifications. The totals can then be reviewed.

# +
from pathlib import Path
import random
import IPython.display as ipd
import time
import pandas as pd


def play_and_classify_m3u(m3u_file, output_file=None, num_tracks=20):
    """
    Play a specified random number of items from an M3U file, ask user to classify as M, D, B, 
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
        classified_df = pd.read_csv(output_file)
        classified_tracks.update(classified_df['Filename'].tolist())
        # Update the classification counts with existing data
        existing_classifications = classified_df['Classification'].value_counts().to_dict()
        for key in ['M', 'D', 'B']:
            if key in existing_classifications:
                classifications[key] += existing_classifications[key]
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

result = {}
for i in ['./00_refined_both_playlist.m3u', './00_refined_dialogue_playlist.m3u', './00_refined_music_playlist.m3u']:
    result[i] = play_and_classify_m3u(i)
