# ML_Audio_ID Project Journal

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

### Day 1

