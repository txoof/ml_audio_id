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
import pandas as pd
import json
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ace_tools_open as tools
# -

timestr = time.strftime('%Y%m%d.%H%M')
timestr

# +
# Load the labeled data from the CSV and JSON files
features_file_path = "./241103_1441_features.csv" 
label_file_path = "./track_classifications_241103.json"
both_file_path = "./00_refined_both_playlist_classification.csv"  # Replace with your actual refined file paths
dialogue_file_path = "./00_refined_dialogue_playlist_classification.csv"
music_file_path = "./00_refined_music_playlist_classification.csv"
training_path = './TRAINING_DATA/PREPARED/outro/'
timestr = time.strftime('%Y%m%d-%HH%MM')
combined_labels_csv_path = f'./combined_labels_{timestr}.csv'

# Load the labeled data from the JSON file
with open(label_file_path, "r") as label_file:
    labels_dict = json.load(label_file)

# Load classification files
both_df = pd.read_csv(both_file_path)
dialogue_df = pd.read_csv(dialogue_file_path)
music_df = pd.read_csv(music_file_path)
labels_df = pd.DataFrame(list(labels_dict.items()), columns=['Filename', 'Label'])

# Load feature data
features_df = pd.read_csv(features_file_path)

# rename columns
for i in [both_df, dialogue_df, music_df]:
    i.rename(columns={'Classification': 'Label'}, inplace=True)


# combine all the tagged files into one consistent dataframe
sources = {
    both_file_path: both_df,
    dialogue_file_path: dialogue_df,
    music_file_path: music_df,
    label_file_path: labels_df
}

classification_map_dict = {
    'B': 'Both',
    'M': 'Music',
    'D': 'Dialogue'
}

# standardize values in the data sources
for file_path, df in sources.items():
    df['Source'] = file_path
    
    # fix filenames that are not appropriately formatted using a mask
    mask = ~df['Filename'].str.startswith(training_path)
    df.loc[mask, 'Filename'] = training_path + df.loc[mask, 'Filename']

    # # ensure that all the files use consistent labeling
    df['Label'] = df['Label'].replace(classification_map_dict).fillna(df['Label'])

# standardize the path in the feature file
features_df['Filename'] = features_df['Filename'].str.replace(r'.*/outro/', training_path, regex=True)

# combine all the sources into a single file
combined_labels_df = pd.concat(list(sources.values()), ignore_index=True)

# find any tracks that have duplicate entries from muliple sources
duplicate_df = combined_labels_df[combined_labels_df.duplicated(keep=False)]
if len(duplicate_df) > 0:
    print(f"{len(duplicate_df)} items found. These items may have conflicting tags. Explore below")
    tools.display_dataframe_to_user(name="Duplicate Rows in DataFrame", dataframe=duplicate_df)
else:
    print(f'No duplicate items found. Ready to proceed with {len(combined_labels_df)} labeled items')
    print(f'Writing combined labels to file: {combined_labels_csv_path}')
    combined_labels_df.to_csv(combined_labels_csv_path, index=False)
# -

# Merge combined_df with features_df on 'Filename'
merged_df = pd.merge(combined_labels_df, features_df, on='Filename', how='inner')
merged_df.to_csv('labeled_tracks_with_features_241106.csv', index=False)

merged_complete = pd.merge(combined_labels_df, features_df, on='Filename', how='right')
merged_complete.to_csv('features_with_filename_some_labels_241106.csv')

# +
# merged_df = pd.read_csv('labeled_tracks_with_features_241106.csv')

# # Drop rows where labels are not found (unlabeled data)
# labeled_data = merged_df.dropna(subset=['Label']).copy()

# # Encode labels as numerical values for classifier training
# label_mapping = {'Music': 0, 'Dialogue': 1, 'Both': 0}
# labeled_data.loc[:, 'Label'] = labeled_data['Label'].map(label_mapping).astype(int)

# # Separate features and labels
# X = labeled_data.drop(columns=['Filename', 'Label', 'Source'])
# y = labeled_data['Label'].astype(int)

# # Check the labels for any issues
# print("Unique labels in y:", y.unique())

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Save the trained model (optional)
# joblib.dump(rf_classifier, 'random_forest_classifier.pkl')
# -


