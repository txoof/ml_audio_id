# ML_Audio_ID Project Journal

## Work Log

[work log](https://docs.google.com/spreadsheets/d/1Cv_RgunlO0N1p7FwOmZY59oMYiBCcdY5pVAlBwqEcnM/edit?usp=sharing)

## Week 0

### Data Preparation

- Setup project and git repo
- Compile training data  & write [`utility scripts`](./utilities/) to init development environment, fetch training data from local server
- create script: [`chunk_mp3.py`](./chunk_mp3.py) to chunk training data into 10 second segments starting from the end to the start
  - The last 10 seconds are the most relevant to the classification of the audio tracks; this contains the "outro" music.
- create script: [`classify_mp3.py`](./classify_mp3.py) to allow manual tagging of training data.

### Pre Reading & Notes

#### [*Efficient Label Collection*](./Wigness_Efficient_Label_Collection_2015_CVPR_paper.pdf)

Short paper on creating efficient label sets from untagged data.

#### [*Hands on Machine Learning*]

Consider implementing online learning to allow consistent training of model and avoid model rot.

- What is the good learning rate for this set?

Chapter 9: Unsupervised Learning

- Investigate using k-means to cluster training data using previously tagged tracks.
- What features to pull from audio?

## Week 1

### 3 November, 2024

#### K-Means Clustering 

Attempted [K-Means Clustering](./kmeans_clustering.ipynb); this yields an 80% solution. About 1/5 of the tracks within the clusters are incorrectly categorized. Other types of clustering will be explored as well.

#### DBSCAN Clustering

Attempted [DBSCAN Clustering](./DBSCAN_clustering.ipynb); this yields very poor results. Everything is classified as "noise". A wide variety of epsilon and neighbor values were tried and no significant results were found.

### 4 November, 2024

#### Hierarchical Cluster Guided Labeling (HCGL)

This method is based on the *Efficient Label Collection* paper. Testing results can be found in [HGCL Clustering](./clustering_HGCL_testing.ipynb). The `ward` type linkage appears to be the most effective at identifying tracks that are of type "music" and "both".

Using a random forest with labeled data to improve accuracy is the next task. Once the classification has been improved, verifying the quality of the classifier is the next step.

### 5 November, 2024

#### Improved Labeling

Used random forrest classifier to identify tracks with highest uncertainty and manually reclassify. After reclassification, tracks from each cluster were manually verified. Results:

**00_refined_both_playlist.m3u**: 4% of the sampled tracks were misclassified as containing music; the remaining 96% were correct though there were several music only tracks classified as containing both dialogue and music (10%).

```Text
Classification Ratios:
M: 0.10
D: 0.04
B: 0.86
Music & Both: 96.0; Dialogue: 4.0
```

**00_refined_dialogue_playlist.m3u**: 0% of the sampled tracks were misclassified.

```Text
Classification Ratios:
M: 0.00
D: 1.00
B: 0.00
Music & Both: 0.0; Dialogue: 100.0
```

**00_refined_music_playlist.m3u**: 0% of the sampled tracks were misclassified.

```Text
Classification Ratios:
M: 1.00
D: 0.00
B: 0.00
Music & Both: 100.0; Dialogue: 0.0
```

#### Training Data Collection

The playlists generated through the improved labeling step can now be used for training. The training sets are now as follows:

- 00_refined_both_playlist.m3u: 510 items containing music or music and dialogue
  - Estimated 96% accurate
- 00_refined_music_playlist.m3u: 157 items containing exclusively music
  - Estimated ~100% accurate
- 00_refined_dialogue_playlist.m3u: 703 items containing exclusively dialogue
  - Estimated ~100% accurate

The training sets for "music" (including mixed music & dialogue) and exclusive "dialogue" are roughly equal in size.

This training data can now be used to develop and evaluate a the best training scheme for identifying audio clips. An additional corpus of ~30K unlabeled items are available for testing.

### 6 November, 2024

#### Combine various labeled sets

Throughout the experimental phase, more and more of the training data was manually labeled. The manually labeled data can be combined to create an improved classifier. This was challenging because the assumptions and techniques used to create the label sets has evolved over through the development process. A stable label set has now been generated and combined with the extracted features.

The data sets are combined with the [combined_label_data](./combine_labeled_data.ipynb) notebook.

#### Develop ensemble classifiers

The ensemble classifier is composed of a random forrest and gradient boosting classifier. The Gradient boosting classifier is weighted more heavily in the soft vote as it is slightly more accurate over all (higher f-1 score). The overall accuracy over the training and sample data is 92% for the ensemble. 

The ensemble classifier can be found in the [classifier_ensemble](./classifier_ensemble.ipynb) notebook.

A manual evaluation of 80 random tracks classified from novel data shows that the classifications were 97.5% accurate for the "Music" and "Both" categories and 97.5% accurate for the "Dialogue" category. This should be acceptable for the purposes of this project.

```
Unique labels in y: [1 0]
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Random Forest Accuracy after Tuning: 0.90

Random Forest Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.89      0.91        44
           1       0.88      0.92      0.90        39

    accuracy                           0.90        83
   macro avg       0.90      0.90      0.90        83
weighted avg       0.90      0.90      0.90        83

Fitting 3 folds for each of 20 candidates, totalling 60 fits
Gradient Boosting Accuracy after Tuning: 0.92

Gradient Boosting Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.91      0.92        44
           1       0.90      0.92      0.91        39

    accuracy                           0.92        83
   macro avg       0.92      0.92      0.92        83
weighted avg       0.92      0.92      0.92        83

Ensemble Classifier Accuracy after Tuning: 0.92

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.91      0.92        44
           1       0.90      0.92      0.91        39

    accuracy                           0.92        83
   macro avg       0.92      0.92      0.92        83
weighted avg       0.92      0.92      0.92        83

```