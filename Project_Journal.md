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

#### Feature Extraction

Determining which features to extract is challenging. I need to learn about different methods for identifying audio features that can differentiate between spoken content and musical content.

I've moved the feature extraction into a the [audio_features](./audio_features.ipynb) notebook to make it easier to re-use this in a standardized way in the future. I've settled on the following features:

Extract the following audio audio features:

- MFCCs: Mel-Frequency Cepstral Coefficients (13 features)
    - widely used in speech and audio processing; measure the power spectrum and capture tonal and timbral features
- ZCR: Zero-Corssing Rate (1 feature)
    - how often the signal amplitude switches across zero; can be used for identifying harmonic features
- Spectral Contrast: mean of each 7 spectral bands (7 features)
    - measures difference in amplitude across different frequencies; music tends to have richer harmonic structure which is reflected in spectral contrast
- Chroma Features: mean of chroma bins (12 features)
    - intensity of the audio signal in 12 pitch classes common in western music; reflects tonal content common in music, but lacking in dialogue
- RMS Energy: Root Mean Square Energy (1 feature)
    - average power of the audio signal over time also known as "loudness"; speech tends to be more dynamic in this domain
- Spectral Rolloff: (1 feature)
    - frequency below which 85% of total spectral energy is contained; music tends to have a broader frequency range

After extracting features, the PCA variance was checked to try to find a cutoff point for the features. The hope was this would help prevent over fit of the data.

![PCA Variance](./assets/PCA_Variance.png)

#### K-Means Clustering

Attempted [K-Means Clustering](./kmeans_clustering.ipynb); this yields midling results. A random sampling of tracks from each cluster shows that clusters 0 and 1 are reasonably well classified as tracks containing some musical elements, but at least 1/5 of all checked samples in both groups is contains only dialogue. Cluster 2 similarly is about 80% only dialogue, but after several rounds of random sampling, at least one track in each sample contains music. This is insufficient for the purposes of this project.

#### t-SNE Clustering

I was dissatisfied with K-Means clustering and decided to try t-SNE clustering. I'm not very familiar with how this works or how to tune it. The results of my fiddling can be seen below.

Experimenting with different cluster sizes yields a value of 3 clusters with a silhouette score of 0.16. This is quite low overall. The t-SNE visualization shows some separation of the clusters, but also considerable overlap.

![t-SNE Clusters](./assets/t-SNE_clusters.png)

#### DBSCAN Clustering

After unsatisfying clustering results using K-Means, I add additional features:

- Spectral Bandwidth:
    - width of spectral range; music typically has a wider bandwidth
- Spectral Centroid:
    - center of mass in the spectrum and brightness of sound; music typically has a hihger brightness
<!-- - Tempo:
    - rythmic structure -->
- HNR: Harmonic-to-Noise-Ratio
    - speech has more harmonic structure

![PCA Variance with Additional Features](./assets/PCA_Variance_additional_features.png)

Based on this graph, I chose to cutoff at 20 components as this contains roughly 80% of the feature data.

Attempted [DBSCAN Clustering](./DBSCAN_clustering.ipynb); this yields very poor results. Everything is classified as "noise". A wide variety of epsilon and neighbor values were tried and no significant results were found. I tried running DBSCAN with a variety of different parameters to try to find a DBSCAN cluster that would yield any significant clustering. 

I primarily varied epsilon over the range of 0.01 to 3 with 30 steps and min_samples over the range 3..9 (inclusive).

Almost all of the samples are identified as noise. Below is a typical clustering output for `eps=0.06157894736842105; min_samples=2`

![Typical DBSCAN Output](./assets/DBSCAN_typical.png)

With my current level of understanding of DBSCAN, this route of clustering data is not worth following any further.

### 4 November, 2024

#### Classification Tool

I need a way to verify the results and to build a labeled set. To help out with that, I built an mp3 player with simple controls that outputs a JSON file with classifications I can use later: [classify_mp3](./classify_mp3.py).

#### Hierarchical Cluster Guided Labeling (HCGL)

This method is based on the *Efficient Label Collection* paper. Testing results can be found in [HGCL Clustering](./clustering_HGCL_testing.ipynb). 

This method was helpful in initially grouping tracks by feature. From here I could create m3u playlists and do some random sampling of the files to verify that the clusters were largely correct. The `ward` and `median` type linkage appears to be the most effective at identifying tracks that are of type "dialogue", "music" and "both".

![Ward Dendrogram](./assets/dendrogram_ward.png)
![Median Dendrogram](./assets/dendrogram_median.png)

I ultimately decided to use the `ward` type with a cutoff at 200100 as it has three distinct groupings. One definite group in the orange segment and two similar groups in the green segment. I believe that this represents the "music" (orange) and "dialogue"/"both" (green) categories.

The cutoff at 200100 should preserve those groupings.

Examining the clusters manually shows that orange cluster contains about 67% "Music" and "Both" and the green cluster contains 67% dialogue. This is reasonably helpful, but not good enough to divide the training sets

```Python
play_and_classify_m3u('orange_median.m3u', num_tracks=1)
Classification Ratios:
M: 0.21
D: 0.33
B: 0.46
Music & Both: 67.3076923076923; Dialogue: 32.69230769230769

play_and_classify_m3u('green_median.m3u', num_tracks=1)
Classification Ratios:
M: 0.00
D: 0.67
B: 0.33
Music & Both: 33.33333333333333; Dialogue: 66.66666666666666
```

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

I've realized that most of the classification was not done through clustering, but rather through iteratively labeling data. My clustering attempts were not terribly successful.

#### Develop ensemble classifiers

The ensemble classifier is composed of a random forrest and gradient boosting classifier. The Gradient boosting classifier is weighted more heavily in the soft vote as it is slightly more accurate over all (higher f-1 score). The overall accuracy over the training and sample data is 92% for the ensemble. 

The ensemble classifier can be found in the [classifier_ensemble](./classifier_ensemble.ipynb) notebook.

A manual evaluation of 80 random tracks classified from novel data shows that the classifications were 97.5% accurate for the "Music" and "Both" categories and 97.5% accurate for the "Dialogue" category. This should be acceptable for the purposes of this project.

```text
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

### 7 November, 2024

#### Next steps

I have experimented heavily with different ensemble models. With some guidance from tutorials and Chat GPT, I've fiddled with hyper parameters and have a pretty successful classifier. It is > 95% accurate with novel data.

I know that I've just grazed the surface of this topic and want to better understand how these algorithms work, and see if there are potentially other algorithms that are more effective that I haven't explored. 

To that end, I've decided to work through the first few chapters of *Hands on Machine Learning with Scikit-Learn , Keras & TensorFlow* (*HOML*). I plan to work some of the sample exercises and see if I can gain some insights on what I have tried up to this point and deepen my understanding.  

The rest of this project period will include a summary of my notes, exercises that I complete and any ideas that I have to improve the work that I have done up to this point.

#### Setting up a Python 3.9.17 Environment

*HOML* has an extensive list of python package requirements; some appear to be pinned to versions that are incompatible with Python 3.12. I narrowed down the latest version of python that should work to 3.9.17, but ran into an issue with creating a virtual environment for that project.

There were various python environment snags that took several hours to unwind. This primarily included diagnosing issues with PyEnv, upgrading various system components and eventually culminating in writing a script to manage virtual environments in the future. 

The script will be useful for setting up future Python projects and can be imported as a git submodule. It can be found at [pyenv_utilities](https://github.com/txoof/pyenv_utilities/blob/main/README.md).

#### Chapter 1

Key points: 

- ML is essentially teaching computers by example from data.
- ML really shines for tasks that are complex and are too difficult to define through hand-crafted rules
- Data is king. More data typically produces higher quality models. Data poverty is a real problem and leads to models that behave erratically. There are lots of ways to fudge slim data sets this that work relatively well like using train-dev data sets that are not 100% representative of production data, but "close enough."

#### Chapter 2 - California Housing Project

The ML Project Checklist:

- [ ] 1. Frame the problem and look at the big picture.
- [ ] 2. Get the data.
- [ ] 3. Explore the data to gain insights
- [ ] 4. Prepare the data to better expose the underlying data patterns to machine learning algorithms.
- [ ] 5. Explore many different models and shortlist the best ones.
- [ ] 6. Fine-tune your models and combine them into a great solution.
- [ ] 7. Present your solution.
- [ ] 8. Launch, monitor, and maintain your sy system.

##### Frame The Problem

- **Objective:** predict a district's median housing price in support a downstream ML system that supports investment decisions.
- **Current State:** Prices are currently estimated manually using complex rules. Predictions can be off by up to 30%
- **Data Set:** Census data contains median housing prices for thousands of districts including other data features.
- **Performance Measure:** Suggested RMSE (root mean square error).
- **Assumptions Check:** E.g. will downstream systems use the exact prices, or bin them into categories? Do calculations need to be precise, or "close enough"?

\[
\text{RMSE}(X, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left(h(x^{(i)}) - {y}^{(i)}\right)^2}
\]
- $m$ is the number of instances of the data set being measured
- $x^{(i)}$ is a vector of all the feature values of the $i^th$ instance in teh data
- $y^{(i)}$ is the label (desired output)
- $X$ is a matrix containing all the feature values (excluding labels); one row per instance and the $i^th$ row is equal to the transpose of $x^{(i)}$
- $h$ is the system prediction function

### 8 November, 2024

For the remaining portion of Chapter 2, all of my work can be found in the [chapter 02 Workbook](./handson_exercises/chapter_02_workbook.ipynb) Jupyter notebook.

#### Chapter 2 Continued

##### Get the Data

Pull the data down as a tarball from github and load into a pandas dataframe.

##### EDA

Take a look at the data and generate some histograms to help with observations

`housing.describe()`:

|       |   longitude |    latitude |   housing_median_age |   total_rooms |   total_bedrooms |   population |   households |   median_income |   median_house_value |
|:------|------------:|------------:|---------------------:|--------------:|-----------------:|-------------:|-------------:|----------------:|---------------------:|
| count | 20640       | 20640       |           20640      |      20640    |        20433     |     20640    |     20640    |     20640       |                20640 |
| mean  |  -119.57    |    35.6319  |              28.6395 |       2635.76 |          537.871 |      1425.48 |       499.54 |         3.87067 |               206856 |
| std   |     2.00353 |     2.13595 |              12.5856 |       2181.62 |          421.385 |      1132.46 |       382.33 |         1.89982 |               115396 |
| min   |  -124.35    |    32.54    |               1      |          2    |            1     |         3    |         1    |         0.4999  |                14999 |
| 25%   |  -121.8     |    33.93    |              18      |       1447.75 |          296     |       787    |       280    |         2.5634  |               119600 |
| 50%   |  -118.49    |    34.26    |              29      |       2127    |          435     |      1166    |       409    |         3.5348  |               179700 |
| 75%   |  -118.01    |    37.71    |              37      |       3148    |          647     |      1725    |       605    |         4.74325 |               264725 |
| max   |  -114.31    |    41.95    |              52      |      39320    |         6445     |     35682    |      6082    |        15.0001  |               500001 |

**Observations**:

- *total_bedrooms* is has rows that are incomplete
- *ocean_proximitiy* is categorical
- *median_house_value* is capped at $500K
- *median_income* is scaled and capped. Values are expressed in units of $10k
- right skew on several features

##### Create a Test Set

Create a test set and hold it in embargo to prevent *data snooping* bias. The best way to do this is to use SciKit's `train_test_split` function. This creates stable sets that are consistent over multiple runs to prevent contaminating the training set with test data and accidentally training the model on test items.

##### Think About Bias

In our problem space, the experts in the downstream forecasting group tell us that median income is very important in predicting median  house prices. *Stratifying* the sample on this might be a good  to ensure that this is captured is captured in the training/test set. Use `pd.cut()` to create a new category that bins median income into five categories.

```Py
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5])
```

*Stratified sampling* can now be done on these groupings.

![Income Category Plot](./assets/homl_ch02_income_plot.png)

SciKit really has you covered. It does it all!

```Py
strat_training_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42
)
```

Comparing the stratified test set versus a randomly split test set.

|   Income Category |   Overall % |   Stratified % |   Random % |   Strat. Error % |   Rand. Error % |
|------------------:|------------:|---------------:|-----------:|-----------------:|----------------:|
|                 1 |        3.98 |           4    |       4.24 |             0.36 |            6.45 |
|                 2 |       31.88 |          31.88 |      30.74 |            -0.02 |           -3.59 |
|                 3 |       35.06 |          35.05 |      34.52 |            -0.01 |           -1.53 |
|                 4 |       17.63 |          17.64 |      18.41 |             0.03 |            4.42 |
|                 5 |       11.44 |          11.43 |      12.09 |            -0.08 |            5.63 |

The differences between the randomized and stratified sets are are significant and meaningful in all the categories.

### 11 November, 2024

#### Yet more Chapter 2

##### Explore and Visualize the Data

Fortunately the housing data has Lat/Lon coordinates, so it's trivial to plot them on an XY scatter plot. The population can be mapped to the dot size and the color can be mapped to the price value.

![Housing Value Map](./assets/homl_ch02_housing_density_value_map.png)

A Pearson Correlation (*r* value) can be calculated for the data as well using the `.corr()` method. These values add depth to the plot. High value houses are highly correlated with median income. So, red dots equal expensive houses for rich people.

|                    |   median_house_value |
|:-------------------|---------------------:|
| median_house_value |            1         |
| **median_income**      |            0.**68838**   |
| total_rooms        |            0.137455  |
| housing_median_age |            0.102175  |
| households         |            0.0714265 |
| total_bedrooms     |            0.0546351 |
| population         |           -0.0201529 |
| longitude          |           -0.0508589 |
| latitude           |           -0.139584  |

![Correlation Matrix Plot](./assets/homl_ch02_corr_matrix_plot.png)

From the plots, it looks like median_income is the best predictor of value. This matches the table. 

![Income vs Value](./assets/homl_ch02_corr_income_value_plot.png)

The income cap that the data preparation team pointed out is really apparent making a horizontal cluster at 500k. There are some other fainter horizontal clusters at 450k and maybe just below 300k. I don't know what causes these.

Doing a little more exploration, creating calculated columns for rooms_per_house, bedroom_ratio and people_per_house yields some interesting results:

|                    |   median_house_value |
|:-------------------|---------------------:|
| **bedrooms_ratio**     |           **-0.256397**  |
| latitude           |           -0.139584  |
| longitude          |           -0.0508589 |
| people_per_house   |           -0.0382244 |
| population         |           -0.0201529 |
| total_bedrooms     |            0.0546351 |
| households         |            0.0714265 |
| housing_median_age |            0.102175  |
| **total_rooms**        |            **0.137455**  |
| **rooms_per_house**    |            **0.143663**  |
| median_income      |            0.68838   |
| median_house_value |            1         |

There's a moderate negative correlation between the value of the house and the bedroom_ratio. As the ratio decreases, the value increases. There's also a weak correlation between the rooms_per_house and value as well as total_rooms and value. As those increase, so does the house value. Those features may be of interest in the training.

#### Prepare the Data for ML Algorithms

##### Clean the Data

ML Algorithms like data that has no gaps. This means that rows containing null values likely need to be dropped or patched up. There are several strategies for doing this. The text suggests using a Scikit simple imputer to fill in the median values.

Median housing values: `[-118.51     34.26     29.     2125.      434.     1167.      408.
    3.5385]`

##### Transforming Categorical & Text Values

Similarly, categorical values like those in the ocean_proximity column need to be encoded as numerical values.

|                 |   count |
|:----------------|--------:|
| ('<1H OCEAN',)  |    7274 |
| ('INLAND',)     |    5301 |
| ('NEAR OCEAN',) |    2089 |
| ('NEAR BAY',)   |    1846 |
| ('ISLAND',)     |       2 |

Finally, those values need to be mapped to column data with binary values. For each unique value in the categorical data, a column will be added. For one-hot column values, the original ocean_proximity will be mapped to a 1 for the column that matches its value (true) and a 0 for all others (false). In the example below, an entity that is ocean_proximity: SPAM, will have the value 00001.

|    |   ocean_proximity_FooBar |   ocean_proximity_Ham |   ocean_proximity_INLAND |   ocean_proximity_NEAR BAY |   ocean_proximity_SPAM |
|---:|-------------------------:|----------------------:|-------------------------:|---------------------------:|-----------------------:|
|  0 |                        0 |                     0 |                        1 |                          0 |                      0 |
|  1 |                        0 |                     0 |                        0 |                          1 |                      0 |
|  2 |                        1 |                     0 |                        0 |                          0 |                      0 |
|  3 |                        0 |                     0 |                        0 |                          0 |                      1 |
|  4 |                        0 |                     1 |                        0 |                          0 |                      0 |

#### Feature Scaling

Most ML algorithms don't perform well when the the input attributes have very different scales. 

**CAUTION:** Only ever scale training data. Train a scaler and then us it transform other data sets including the validation set! 

I'm not sure why this caution exists, but the author makes a big deal of this.

Some strategies:

* Min-Max (normalizatio) scaling: map the values, in whatever their range, to floats between 0 and 1 or -1 and 1 (useful for Neural Networks that prefer zero mean data)
  * I imagine that if you have extreme outliers, this will totally fubar up your scaling
  * Skewed data with a long tail impacts this method 
* Standardization: subtracts the mean value pushing values close to the mean to zero and then divides the result by the standard deviation. 
  * This is less impacted by outliers, but long tail skew can squash out the bulk of the data into a small range

For both of these methods, transform it to shrink the tail and make the distribution symmetrical where possible.

We can take advantage of SciKit's StandardScalar transformer to do this reliably and shift all the values to a range between -2 and 2.

Numerical Values from housing data prior to scaling:

|    |   longitude |   latitude |   housing_median_age |   total_rooms |   total_bedrooms |   population |   households |   median_income |
|---:|------------:|-----------:|---------------------:|--------------:|-----------------:|-------------:|-------------:|----------------:|
|  0 |     -122.42 |      37.8  |                   52 |          3321 |             1115 |         1576 |         1034 |          2.0987 |
|  1 |     -118.38 |      34.14 |                   40 |          1965 |              354 |          666 |          357 |          6.0876 |
|  2 |     -121.98 |      38.36 |                   33 |          1083 |              217 |          562 |          203 |          2.433  |
|  3 |     -117.11 |      33.75 |                   17 |          4174 |              851 |         1845 |          780 |          2.2618 |
|  4 |     -118.15 |      33.77 |                   36 |          4366 |             1211 |         1912 |         1172 |          3.5292 |
|  5 |     -121.31 |      37.96 |                   52 |          1829 |              301 |          694 |          319 |          3.3466 |
|  6 |     -118.36 |      33.85 |                   34 |          1086 |              197 |          509 |          158 |          6.1133 |
|  7 |     -122.13 |      37.41 |                   36 |          4787 |              900 |         2039 |          890 |          5.4063 |
|  8 |     -118.44 |      34.31 |                   22 |          3182 |              822 |         2661 |          746 |          2.7472 |
|  9 |     -118.13 |      34.06 |                   17 |          1714 |              572 |         1590 |          568 |          1.1875 |

After scaling (as np array):

```TXT
[-1.42303652  1.0136059   1.86111875  0.31191221  1.36816703  0.13746004   1.39481249 -0.93649149]
[ 0.59639445 -0.702103    0.90762971 -0.30861991 -0.43592476 -0.69377062  -0.37348471  1.17194198]
[-1.2030985   1.27611874  0.35142777 -0.71224036 -0.76070869 -0.78876841  -0.77572662 -0.75978881]
[ 1.23121557 -0.88492444 -0.91989094  0.70226169  0.74230601  0.38317548   0.73137454 -0.85028088]
[ 0.71136206 -0.87554898  0.58980003  0.79012465  1.59575285  0.44437597   1.75526303 -0.18036472]
[-0.86819286  1.08860957  1.86111875 -0.37085617 -0.5615711  -0.66819429  -0.47273921 -0.27688255]
[ 0.60639163 -0.83804715  0.43088519 -0.7108675  -0.8081224  -0.83718074  -0.89326484  1.18552636]
[-1.27807737  0.83078446  0.58980003  0.98278248  0.85846961  0.56038289   1.01869019  0.81182372]
[ 0.5664029  -0.6224116  -0.52260385  0.24830309  0.67355613  1.12854275   0.64256789 -0.5937105 ]
[ 0.72135924 -0.73960483 -0.91989094 -0.42348242  0.08088472  0.1502482    0.17763893 -1.41812918]
```

### 12 November, 2024

#### Custom Transformers

If you don't find just the right transformer, you can write your own class. As long as the class suports `fit` and `transform` methods, it will work. This is great if you really know what you're looking for because you can use pretty much functions or methods you can get your hands on. The diagram below (borrowed from the author's notebooks) is generated using a custom built transformer class. I don't entirely understand how it works, but I see the general outlines of how it was built. It's also possible to specify inverse functions so it's easy to work with the predictions and match them with the labeled data.

```Python
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

![k-Means Clustering](./assets/homl_ch02_custom_transformer_k-means.png)

#### Transformation Pipelines

Processing data can potentially take tens of steps to prepare prior to doing any sort of classification task. Fortunately, SciKit has a `pipeline` class that allows defining a set of steps to take on a particular feature set. It's as easy as defining the variable and adding an array of tuples that define a name and each step.

I have to assume that this is working as expected because a lot of this is over my head. My results match the author's, but that's be best confirmation I can arrive at at this point.

```Python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
num_pipeline
```

SciKit also provides a great pipeline visualization tool that allows easy inspection of pipeline objects.

```Python
import sklearn
sklearn.set_config(display="diagram")
```

![num_pipeline diagram](./assets/homl_ch02_pipeline_diagram.png)

The code below was borrowed from the author's notebooks, pulled together to show the entire process and annotated to show the various steps that do the following steps. Several data pipeline steps are specifically noted [A..E].

- Download the dataset
- add the `income_cat` feature and then split the data into stratified train/test sets
- remove the `income_cat` column
- Create pipeline with the following properties
  - (A) impute missing values
  - (B) encode categorical data as binary one-hot columns
  - (C\) create ratio features
  - (D) cluster similarity features
  - (E) transform "long-tail" data into more gaussian (normal) distributions

```Python
from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler #, OrdinalEncoder
from sklearn.model_selection import train_test_split



from sklearn.cluster import KMeans

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"), # (A) impute missing values
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), # (C) Create ratio features
        StandardScaler()) # (F) scale all the values

# load unprocessed data
housing = load_housing_data()

# add an income_cat column
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5])
# split into stratified train/test split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42)

# drop the income-cat column
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing_labels = strat_train_set["median_house_value"].copy()

# WHOA! PAY ATTENTION HERE!
housing = strat_train_set.drop("median_house_value", axis=1)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), # (A) impute missing values
    OneHotEncoder(handle_unknown="ignore")) # (B) encode categorical data as binary one-hot columns


# (E) transform "long-tail" data into more gaussian (normal) distributions
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"), #(A) impute missing values
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()) # (F) scale all the values
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), # (A) impute missing values
                                     StandardScaler()) # (F) scale all the values
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age
```

#### Feedback for the Author

Pulling this together was really challenging because the code is scattered throughout the book and all over the provided notebook. I find the author's inconsistent use of variable names (reassignments without note) very challenging as well. You'll notice that in the example above, `housing` starts its life as the complete unprocessed data pulled from a tarball. Then `housing` is split into stratified train/test sets. A few steps later, housing is reassigned to reference the training set. 

An incautious read of the code would lead you to believe that the raw housing data is being fed directly into the pipeline and prepared for training in the later parts of the chapter. That's not the case! On line 85 of the code above (`WHOA! PAY ATTENION`), `housing` is reassigned.

I've written a few pieces of feedback for the author on this point. See issues [#167](https://github.com/ageron/handson-ml3/issues/167) and [#168](https://github.com/ageron/handson-ml3/issues/168) on GitHub for the full suggestion.

#### Chapter 2 Pause

I'm pretty frustrated at this point and feel like a need a break. I'm going to move over to some other learning tasks that I have been meaning to do for some time. I might loop back around to this again later in the week once my frustration subsides a bit.


#### SQL

I've decided to take a bit of a break from the *HOML* book and move on to completing some SQL courses on DataCamp. My knowledge of SQL up to this point is primarily of PL/SQL and it's all self taught through necessity at a previous job. We were implementing D2L Brightspace on our campus and needed to connect it to our backend Student Information System (SIS) so we could provide students and teachers appropriate access to their courses. 

The SIS only spoke PL/SQL and I was the only person who was trained on D2L Admin, so I quickly picked up some skills. I think I probably have a lot of bad habits and gaps in my understanding. I'm going to use the next few days to bolster my SQL skills.

##### Introduction to SQL

![Intro to SQL Certificate](./assets/DataCamp_intro_to_SQL.pdf.png)

This was all good review. Nothing new in this section for me, but it's reassuring to know that I don't have any major gaps in my knowledge when it comes to the basics.

##### Intermediate SQL

![Intermediate SQL](./assets/DataCamp_intermediate_SQL.pdf.png)

This section was also excellent review. The review of the aggregation and grouping functions was really useful. I never learned that very well. 

##### Joining Data in SQL

![Joining Data in SQL](./assets/DataCamp_intermediate_SQL.pdf.png)

This was excellent. I had no idea there was anything beyond left, right and inner joins. Learning about cross and self joins was helpful. The subquery practice was also super helpful. I've borrowed from stack overflow plenty with sub queries, but now I feel confident that I can apply this skill in other contexts.