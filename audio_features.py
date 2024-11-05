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
import random
import IPython.display as ipd
import time

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


# -

def write_m3u(cluster, output_file):
    """
    Write an M3U playlist file for the given cluster.
    
    Parameters:
    cluster (pd.DataFrame): DataFrame containing the filenames of the cluster.
    output_file (str or Path): Path to the output M3U file.
    """
    output_file = Path(output_file)
    if not output_file.name.endswith('.m3u'):
        output_file = Path(output_file.name + '.m3u')
        
    with open(output_file, 'w') as f:
        f.write('#EXTM3U\n')
        for filename in cluster['Filename']:
            f.write(f"{filename}\n")


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

# +
# from IPython.display import HTML

# def create_audio_player_with_controls(file_path):
#     """
#     create an embedded audio player that allows ff/rewind using left and right arrows
#     """
#     audio_html = f"""
#     <audio id="audioPlayer" controls autoplay>
#       <source src="{file_path}" type="audio/mp3">
#       Your browser does not support the audio element.
#     </audio>
#     <br>
#     <button onclick="document.getElementById('audioPlayer').currentTime += 2">Skip Forward 2s</button>
#     <button onclick="document.getElementById('audioPlayer').currentTime -= 2">Skip Backward 2s</button>
#     <script>
#       document.addEventListener('keydown', function(event) {{
#         var audio = document.getElementById('audioPlayer');
#         if (event.code === 'ArrowRight') {{
#           audio.currentTime += 2;  // Skip forward 2 seconds
#         }}
#         if (event.code === 'ArrowLeft') {{
#           audio.currentTime -= 2;  // Skip backward 2 seconds
#         }}
#       }});
#     </script>
#     """
#     return HTML(audio_html)

# +
# import random
# import IPython.display as ipd
# import time
# def play_and_classify_m3u(m3u_file, output_file=None, num_tracks=20):
#     """
#     Play a specified or random number of items from an M3U file, ask user to classify as M, D, B, 
#     and record the values in a file, displaying the ratio of M, D, B.
#     Press 'Q' to quit at any time.
    
#     Parameters:
#     m3u_file (str or Path): Path to the M3U file.
#     output_file (str or Path): Path to the output file where classifications are saved.
#     num_tracks (int, optional): Number of tracks to play. If None, a random number of tracks will be played.
#     """
#     m3u_file = Path(m3u_file)

#     if not output_file:
#         output_file = m3u_file.parent / (m3u_file.stem + '_classification.csv')
#     else:
#         output_file = Path(output_file)
    
#     # Read M3U file
#     with m3u_file.open('r') as f:
#         lines = f.readlines()
    
#     # Filter lines to get only filenames (skip #EXTM3U and other comments)
#     tracks = [line.strip() for line in lines if not line.startswith('#')]
    
#     # Read existing classifications if the output file exists
#     classified_tracks = set()
#     classifications = {'M': 0, 'D': 0, 'B': 0}
#     if output_file.exists():
#         classified_df = pd.read_csv(output_file)
#         classified_tracks.update(classified_df['Filename'].tolist())
#         # Update the classification counts with existing data
#         existing_classifications = classified_df['Classification'].value_counts().to_dict()
#         for key in ['M', 'D', 'B']:
#             if key in existing_classifications:
#                 classifications[key] += existing_classifications[key]
#     else:
#         with output_file.open('w') as out_file:
#             out_file.write('Filename,Classification\n')
    
#     # Filter out tracks that have already been classified
#     tracks_to_play = [track for track in tracks if track not in classified_tracks]
    
#     # Randomly shuffle the list of tracks to play
#     random.shuffle(tracks_to_play)
    
#     # Determine the number of tracks to play
#     num_tracks = min(num_tracks, len(tracks_to_play))
    
#     # Open output file for appending classifications
#     with output_file.open('a') as out_file:
#         # Play each track and ask for classification
#         for i in range(num_tracks):
#             track = tracks_to_play[i]
#             print(f"Playing track {i + 1} of {num_tracks}: {track}")
            
#             # Automatically play the audio file
#             audio = ipd.Audio(track, autoplay=True)
#             display_handle = ipd.display(audio, display_id=True)
            
#             # Get user classification (M, D, B, or Q to quit)
#             classification = None
#             while classification not in ['M', 'D', 'B', 'Q']:
#                 classification = input("Classify this track as M (Music), D (Dialogue), B (Both), or Q (Quit): ").upper()
#                 if classification == 'Q':
#                     print("Quitting...")
#                     return
            
#             # Record classification if not quitting
#             if classification != 'Q':
#                 classifications[classification] += 1
#                 out_file.write(f"{track},{classification}\n")
            
#             # Pause between tracks
#             time.sleep(1)
#             display_handle.update(ipd.HTML(''))
    
#     # Calculate and display ratio of classifications
#     total_classifications = sum(classifications.values())
#     print("\nClassification Ratios:")
#     for key, value in classifications.items():
#         ratio = value / total_classifications if total_classifications > 0 else 0
#         print(f"{key}: {ratio:.2f}")
#     dialog_pct = classifications["D"] / total_classifications
#     music_both_pct = (classifications["M"] + classifications["B"]) / total_classifications
#     print(f"Music & Both: {music_both_pct * 100}; Dialogue: {dialog_pct * 100}")
#     return classifications
