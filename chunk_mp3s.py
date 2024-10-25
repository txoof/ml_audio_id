#!/usr/bin/env python3
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

from pathlib import Path
import logging
import json
from datetime import datetime
import argparse


# +
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3

from mutagen.id3 import ID3NoHeaderError
from mutagen import MutagenError
from pydub.exceptions import CouldntDecodeError
# -

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO, log_to_file=False, log_dir="logs"):
    """
    Set up logging configuration for the project.

    Parameters:
    log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    log_to_file (bool): If True, logs will also be written to a file.
    log_dir (str or Path): Directory where the log file will be saved if `log_to_file` is True.
    """
    # Set up log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure the root logger
    logging.basicConfig(level=log_level, format=log_format)
    
    if log_to_file:
        # Ensure log directory exists
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler
        file_handler = logging.FileHandler(log_dir / "project.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)



def get_timestamp():
    """
    Returns the current date and time as a string in the format YYYY.MM.DD-HH.MM.
    """
    return datetime.now().strftime("%Y.%m.%d-%H.%M")



def locate_mp3_files(root_dir, glob="*.mp3"):
    """
    Recursively locate all MP3 files within a given directory path.

    Parameters:
    root_dir (str or Path): The root directory to start the search.
    glob (str): The file pattern to match (default is '*.mp3').

    Returns:
    list of Path: A list of Path objects pointing to each MP3 file found.
    """
    mp3_files = []
    root_path = Path(root_dir)
    
    try:
        if not root_path.exists():
            logger.error(f"The specified directory does not exist: {root_path}")
            return mp3_files
        
        if not root_path.is_dir():
            logger.error(f"The specified path is not a directory: {root_path}")
            return mp3_files

        # Perform the search for files matching the glob pattern
        mp3_files = list(root_path.rglob(glob))
        logging.info(f"Located {len(mp3_files)} MP3 files in '{root_path}'.")

    except PermissionError:
        logging.error(f"Permission denied while accessing directory: {root_path}")
    except FileNotFoundError:
        logging.error(f"The directory was not found: {root_path}")
    except OSError as e:
        logging.error(f"OS error occurred while accessing '{root_path}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error while locating MP3 files in '{root_path}': {e}")

    return mp3_files


def split_mp3_files_to_chunks(mp3_files, output_dir, chunk_duration=10):
    """
    Splits each MP3 file in the provided list into 10-second chunks starting from the end of the file.
    
    Parameters:
    mp3_files (list of Path): List of Path objects for MP3 files to be split.
    output_dir (str or Path): Directory where chunks will be saved.
    chunk_duration (int): Duration of each chunk in seconds (default is 10).
    
    Returns:
    dict: A dictionary mapping each original file to a list of its chunked output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Chunking {len(mp3_files)} files')

    chunk_dict = {}
    
    for mp3_file in mp3_files:
        try:
            logger.debug(f'Processing file: {mp3_file}')
            try:
                audio = AudioSegment.from_mp3(mp3_file)
            except CouldntDecodeError:
                logger.error(f"Could not decode MP3 file: {mp3_file}. Skipping file.")
                continue

            duration_ms = len(audio)
            chunk_duration_ms = chunk_duration * 1000

            # Load ID3 tags from the original file
            try:
                tags = {key: value[0] for key, value in EasyID3(mp3_file).items()}
            except ID3NoHeaderError:
                logger.warning(f"No ID3 header found in file: {mp3_file}. Skipping ID3 tag copying.")
                tags = {}
            except MutagenError as e:
                logger.error(f"Error loading ID3 tags from file: {mp3_file}. Error: {e}")
                continue

            logger.debug(f'Tags: {tags}')

            # Create chunks starting from the end of the file
            chunk_start = max(0, duration_ms - chunk_duration_ms)
            chunk_num = 1

            chunk_dict[mp3_file] = []

            while chunk_start >= 0:
                chunk_end = chunk_start + chunk_duration_ms
                chunk = audio[chunk_start:chunk_end]

                # Generate a unique output file name for each chunk
                chunk_filename = f"{mp3_file.stem}_chunk_{chunk_num}.mp3"
                chunk_output_path = output_dir / chunk_filename

                # Export chunk and add original ID3 tags
                try:
                    chunk.export(chunk_output_path, format="mp3", tags=tags)
                    chunk_dict[mp3_file].append(chunk_output_path)
                    logger.debug(f"Created chunk {chunk_num} for file {mp3_file}")
                except Exception as e:
                    logger.error(f"Error exporting chunk {chunk_num} for file {mp3_file}. Error: {e}")
                    break

                chunk_num += 1
                chunk_start -= chunk_duration_ms

            logger.info(f"Finished processing file: {mp3_file}")

        except Exception as e:
            logger.error(f"Unexpected error processing file {mp3_file}: {e}")

    return chunk_dict


def load_processed_files(tracking_file):
    """
    Loads the list of processed files from a JSON file.
    
    Parameters:
    tracking_file (str or Path): Path to the JSON file storing processed files.

    Returns:
    dict: A dictionary with file paths as keys and timestamps as values.
    """
    tracking_file = Path(tracking_file)
    processed_files = {}
    
    if tracking_file.exists():
        try:
            with tracking_file.open('r') as f:
                processed_files = json.load(f)
                # Convert to dictionary format if it is a list of paths only
                if isinstance(processed_files, list):
                    processed_files = {entry: "" for entry in processed_files}
        except json.JSONDecodeError:
            logger.error(f"JSON decoding error for tracking file {tracking_file}. File may be corrupted.")
        except Exception as e:
            logger.error(f"Error loading tracking file {tracking_file}: {e}")
    
    return processed_files


def save_processed_files(processed_files, tracking_file):
    """
    Saves the list of processed files with timestamps to a JSON file.

    Parameters:
    processed_files (dict): Dictionary with file paths as keys and timestamps as values.
    tracking_file (str or Path): Path to the JSON file to store processed files.

    Returns:
    None
    """
    tracking_file = Path(tracking_file)
    try:
        with tracking_file.open('w') as f:
            json.dump(processed_files, f, indent=4)
    except IOError:
        logger.error(f"Unable to write to tracking file {tracking_file}. Check file permissions.")
    except Exception as e:
        logger.error(f"Unexpected error saving tracking file {tracking_file}: {e}")


def update_processed_files(new_files, tracking_file):
    """
    Loads processed files, updates with new files and timestamps, and saves back to disk.

    Parameters:
    new_files (list of Path): List of new files to mark as processed.
    tracking_file (str or Path): Path to the JSON file storing processed files.

    Returns:
    dict: Updated dictionary of processed files with timestamps.
    """
    processed_files = load_processed_files(tracking_file)
    timestamp = get_timestamp()
    
    # Update processed files with new files and current timestamp
    processed_files.update({str(file): timestamp for file in new_files})
    
    save_processed_files(processed_files, tracking_file)
    return processed_files


def prune_processed_files(mp3_files, processed_files):
    """
    Prunes the list of mp3 files to remove files that are already in the processed files dictionary.

    Parameters:
    mp3_files (list of Path): List of Path objects representing MP3 files to process.
    processed_files (dict): Dictionary of file paths (str) as keys and timestamps as values.

    Returns:
    list of Path: A pruned list of MP3 files that excludes already processed files.
    """
    processed_paths = set(processed_files.keys())
    return [file for file in mp3_files if str(file) not in processed_paths]


def parse_arguments():
    """
    Parse command-line arguments for the main function parameters.

    Returns:
    argparse.Namespace: Parsed arguments as namespace object.
    """
    parser = argparse.ArgumentParser(description="Process MP3 files by splitting them into 10-second chunks.")
    parser.add_argument('source_dir', type=str, help="Directory containing MP3 files to process.")
    parser.add_argument('--output_dir', type=str, default='./output', help="Directory where chunks will be saved.")
    parser.add_argument('--tracking_file', type=str, default='./processed_tracks.json', help="Path to the JSON file tracking processed files.")
    parser.add_argument('--chunk_duration', type=int, default=10, help="Duration of each chunk in seconds.")
    
    return parser.parse_args()


def main(source_dir='', output_dir='./output', tracking_file='./processed_tracks.json', chunk_duration=10):
    
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    setup_logging(logging.INFO)
    
    processed_files = load_processed_files(tracking_file)
    source_mp3_files = locate_mp3_files(source_dir)
    unprocessed_files = prune_processed_files(source_mp3_files, processed_files)

    if unprocessed_files:
        logging.info(f'Processing {len(unprocessed_files)} new files')
        processed_chunks = split_mp3_files_to_chunks(unprocessed_files, output_dir)

        update_processed_files(unprocessed_files, tracking_file)   
    else:
        logging.info(f'No new files were found in {source_dir}')
    logging.info('Done')


if __name__ == "__main__":
    args = parse_arguments()
    main(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        tracking_file=args.tracking_file,
        chunk_duration=args.chunk_duration
    )
