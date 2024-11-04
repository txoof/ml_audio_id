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
from pathlib import Path
import librosa
import numpy as np
import logging
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Configure logging for importable module
logger = logging.getLogger(__name__)


# -

def extract_features(file_path):
    """
    Extracts audio features from a given audio file.

    Parameters:
    file_path (Path): Path to the audio file.

    Returns:
    numpy.ndarray: Feature vector containing consistently reduced audio features.
    """
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs (mean and standard deviation of each coefficient across time)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Extract Zero-Crossing Rate (mean)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Extract Spectral Contrast (mean and standard deviation)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)

    # Extract Chroma Features (mean and standard deviation)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Extract Spectral Rolloff (mean)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    # Extract Spectral Centroid (mean)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Extract RMS Energy (mean)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # Combine selected features into a single feature vector
    features = np.concatenate((
        mfccs_mean,
        mfccs_std,
        [zcr_mean],
        spectral_contrast_mean,
        spectral_contrast_std,
        chroma_mean,
        chroma_std,
        [spectral_rolloff_mean],
        [spectral_centroid_mean],
        [rms_mean]
    ))

    return features



p = Path('./TRAINING_DATA/')

isinstance(p, str)


# Extract features from all audio files
def extract_features_from_directory(audio_dir):
    """
    Extracts features from all audio files in a given directory using parallel processing to utilize multiple cores.

    Parameters:
    audio_dir (Path or str): Path to the directory containing audio files.

    Returns:
    tuple: A tuple containing a numpy array of feature vectors and a list of filenames.
    """

    if not isinstance(audio_dir, Path):
        audio_dir = Path(audio_dir)
    
    filenames = [filename for filename in audio_dir.iterdir() if filename.suffix == '.mp3']
    
    logging.info(f'Found {len(filenames)} audio files in {audio_dir}')
    
    # Use tqdm to show progress indicator
    feature_list = Parallel(n_jobs=-1)(delayed(extract_features)(filename) for filename in tqdm(filenames, desc='Extracting features'))
    
    return np.array(feature_list), filenames


# +
def write_features_to_file(features, filenames, output_file):
    """
    Writes the extracted features and corresponding filenames to a file.

    Parameters:
    features (np.ndarray): Array of feature vectors.
    filenames (list): List of filenames corresponding to the features.
    output_file (str): Path to the output file.
    """
    df = pd.DataFrame(features)
    df['Filename'] = filenames
    df.to_csv(output_file, index=False)
    logger.info(f'Features written to {output_file}')

def read_features_from_file(input_file):
    """
    Reads features and filenames from a file.

    Parameters:
    input_file (str): Path to the input file containing features and filenames.

    Returns:
    tuple: A tuple containing a numpy array of feature vectors and a list of filenames.
    """
    df = pd.read_csv(input_file)
    filenames = df['Filename'].tolist()
    features = df.drop(columns=['Filename']).values
    logger.info(f'Features read from {input_file}')
    return features, filenames
