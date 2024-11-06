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

# +
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

features_labels_file = './features_with_filename_some_labels_241106.csv'

merged_df = pd.read_csv(features_labels_file)

# Drop rows where labels are not found (unlabeled data)
labeled_data = merged_df.dropna(subset=['Label']).copy()
unlabeled_data = merged_df[merged_df['Label'].isna()].copy()

# Encode labels as numerical values for classifier training
label_mapping = {'Music': 0, 'Dialogue': 1, 'Both': 0}
labeled_data.loc[:, 'Label'] = labeled_data['Label'].map(label_mapping).astype(int)

# Separate features and labels
X = labeled_data.drop(columns=['Filename', 'Label', 'Source'])
y = labeled_data['Label'].astype(int)

# Check the labels for any issues
print("Unique labels in y:", y.unique())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest using RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist_rf, n_iter=20, cv=3, n_jobs=-1, verbose=1, random_state=42)
random_search_rf.fit(X_train, y_train)

# Get the best estimator from the grid search for Random Forest
best_rf_classifier = random_search_rf.best_estimator_

# Make predictions on the test set using the Random Forest
y_pred_rf = best_rf_classifier.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy after Tuning: {accuracy_rf:.2f}")
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Hyperparameter tuning for Gradient Boosting using RandomizedSearchCV
param_dist_gb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}

gb_classifier = GradientBoostingClassifier(random_state=42)
random_search_gb = RandomizedSearchCV(estimator=gb_classifier, param_distributions=param_dist_gb, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search_gb.fit(X_train, y_train)

# Get the best estimator from the grid search for Gradient Boosting
best_gb_classifier = random_search_gb.best_estimator_

# Make predictions on the test set using the Gradient Boosting classifier
y_pred_gb = best_gb_classifier.predict(X_test)

# Evaluate the Gradient Boosting model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy after Tuning: {accuracy_gb:.2f}")
print("\nGradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))

# Create an ensemble using VotingClassifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', best_rf_classifier),
    ('gb', best_gb_classifier)
], voting='soft', weights=[1, 2])

# Train the ensemble model
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set using the ensemble model
y_pred = voting_classifier.predict(X_test)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Classifier Accuracy after Tuning: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# +
# Predict labels for the unlabeled data
X_unlabeled = unlabeled_data.drop(columns=['Filename', 'Label', 'Source'], errors='ignore')
y_unlabeled_pred = voting_classifier.predict(X_unlabeled)

# Add predictions to the unlabeled data
unlabeled_data['Predicted_Label'] = y_unlabeled_pred

# Map predicted labels back to original categories
reverse_label_mapping = {0: 'Music', 1: 'Dialogue'}
unlabeled_data['Predicted_Label'] = unlabeled_data['Predicted_Label'].map(reverse_label_mapping)

# Create m3u playlists for the predicted labels
with open('music_both.m3u', 'w') as music_playlist:
    music_playlist.write('#EXTM3U\n')
    for _, row in unlabeled_data.iterrows():
        if row['Predicted_Label'] == 'Music':
            music_playlist.write(f"{row['Filename']}\n")

with open('dialogue.m3u', 'w') as dialogue_playlist:
    dialogue_playlist.write('#EXTM3U\n')
    for _, row in unlabeled_data.iterrows():
        if row['Predicted_Label'] == 'Dialogue':
            dialogue_playlist.write(f"{row['Filename']}\n")
# -
# Save the trained models (optional)
joblib.dump(best_rf_classifier, 'random_forest_classifier.pkl')
joblib.dump(best_gb_classifier, 'gradient_boosting_classifier.pkl')
joblib.dump(voting_classifier, 'voting_classifier.pkl')
# +
import random
import IPython.display as ipd
import time
from pathlib import Path
from audio_features import read_classifications

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
    classification_map = {
        'M': 'Music',
        'B': 'Both',
        'D': 'Dialogue',
        'Q': 'Quit',
    }

    classifications = {
        'B': 0,
        'M': 0,
        'D': 0
    }
    
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
            while classification not in classification_map.keys():
                classification = input("Classify this track as M (Music), D (Dialogue), B (Both), or Q (Quit): ").upper()
                if classification == 'Q':
                    print("Quitting...")
                    return
            
            # Record classification if not quitting
            if classification != 'Q':
                classifications[classification] += 1
                out_file.write(f"{track},{classification_map[classification]}\n")
            
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


# +
play_and_classify_m3u('./dialogue.m3u')

play_and_classify_m3u('./music_both.m3u')
# -

print(read_classifications('./dialogue_classification.csv'))
print(read_classifications('./music_both_classification.csv'))


