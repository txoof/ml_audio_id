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
import argparse
import time
import sys

try:
    from pydub import AudioSegment
    from datetime import datetime
except Exception:
    print('This script must be run in the local virtual environment. Exiting.')
    sys.exit(-1)


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
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "%y%m%d %H%M.%S"  # Custom date format YY.MM.DD-HHMM.SS
    
    # Configure the root logger
    logging.basicConfig(level=log_level, format=log_format, datefmt=datefmt)
    
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

    logging.info(f'searching {root_dir} for {glob}')
    
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

    logging.info(f'found {len(mp3_files)} matching files')
    return mp3_files


# +
def process_mp3_files(mp3_files, output_dir="output_chunks", log_file="processed_files.json", chunk_length=10000):
    """
    Process a list of MP3 files, splitting each into segments of chunk_length from end to start.

    Parameters:
    mp3_files (list of Path): List of MP3 files to process.
    output_dir (str or Path): Directory to save the output chunks.
    log_file (str or Path): Path to the JSON log file for processed files.
    chunk_length (int): Length of each audio chunk in milliseconds (default: 10000).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_data = load_processed_data(log_file)

    for file_path in mp3_files:
        logging.debug(f'processing {file_path}')
        file_name = file_path.name

        # Skip files marked as successfully processed
        if file_name in processed_data and processed_data[file_name]["status"] == "success":
            logging.info(f"Skipping previously processed file: {file_name}")
            continue

        try:
            # Load the MP3 file
            try:
                audio = AudioSegment.from_mp3(file_path)
            except FileNotFoundError:
                logging.error(f"File not found: {file_name}")
                record_failure(processed_data, file_name, "FileNotFoundError", "File not found")
                continue
            except Exception as e:
                logging.error(f"Failed to load MP3 file {file_name}: {e}")
                record_failure(processed_data, file_name, "LoadError", str(e))
                continue

            # Calculate and export chunks
            duration = len(audio)
            chunk_start = max(duration - 10000, 0)
            segment_index = 1

            while chunk_start >= 0:
                chunk = audio[chunk_start:chunk_start + 10000]
                chunk_output_path = output_dir / f"{file_path.stem}_chunk_{segment_index}.mp3"
                try:
                    chunk.export(chunk_output_path, format="mp3")
                except PermissionError:
                    logging.error(f"Permission denied when exporting chunk for {file_name}")
                    record_failure(processed_data, file_name, "PermissionError", "Permission denied for chunk export")
                    break
                except Exception as e:
                    logging.error(f"Failed to export chunk for {file_name}: {e}")
                    record_failure(processed_data, file_name, "ExportError", str(e))
                    break
                segment_index += 1
                chunk_start -= 10000

            # Log successful processing if all chunks were saved
            if chunk_start < 0:
                processed_data[file_name] = {
                    "timestamp": get_timestamp(),
                    "status": "success"
                }
                logging.info(f"Successfully processed file: {file_name}")

        except Exception as e:
            logging.error(f"Unexpected error during processing of {file_name}: {e}")
            record_failure(processed_data, file_name, "UnexpectedError", str(e))

        # Update log file after each file
        save_processed_data(log_file, processed_data)


def record_failure(processed_data, file_name, error_type, error_message):
    """
    Record a failed processing attempt in the processed data log.

    Parameters:
    processed_data (dict): Dictionary containing processed file records.
    file_name (str): Name of the file that failed processing.
    error_type (str): Type of error encountered.
    error_message (str): Description of the error encountered.
    """
    logging.debug(f'recording failure for: {file_name}')
    processed_data[file_name] = {
        "timestamp": get_timestamp(),
        "status": "failure",
        "error_type": error_type,
        "error_message": error_message
    }


def load_processed_data(log_file):
    """
    Load the JSON log file of processed files.

    Parameters:
    log_file (str or Path): Path to the JSON log file.

    Returns:
    dict: Dictionary with processed file information.
    """
    log_path = Path(log_file)
    if log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return {}


def save_processed_data(log_file, data):
    """
    Save the processed file data to a JSON log file.

    Parameters:
    log_file (str or Path): Path to the JSON log file.
    data (dict): Dictionary with processed file information to save.
    """
    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)


# -

def parse_arguments():
    """
    Parse command-line arguments for the MP3 file processing script.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process MP3 files into specified-length chunks from end to start.")
    parser.add_argument("-v", action="count", default=0, help="Increase logging verbosity (e.g., -v, -vv, -vvv)")
    parser.add_argument("-l", "--log_path", type=str, help="Directory for log file output if logging to file")
    parser.add_argument("--log_to_file", action="store_true", help="Log output to a file instead of screen (default: screen)")
    parser.add_argument("-s", "--source", type=str, default="INPUT", help="Root directory to search for MP3 files")
    parser.add_argument("-o", "--output", type=str, default="OUTPUT", help="Directory to save output chunks")
    parser.add_argument("-p", "--processed_log", type=str, default="processed_files.json", help="JSON file to log processed files")
    parser.add_argument("--chunk_length", type=int, default=10, help="Length of each audio chunk in seconds (default: 10)")
    parser.add_argument("-y", "--summary", action="store_true", help="Show a summary of how many files were processed")
    parser.add_argument("-f", "--show_failures", action="store_true", help="Show a list of all files that failed processing")
    return parser.parse_args()



def main():
    """
    Main function to initiate MP3 file processing, splitting them into specified-length segments,
    and logging the results.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level based on verbosity count (-v)
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.v, len(log_levels) - 1)]
    
    # Set up logging based on user choice (file or screen)
    log_dir = args.log_path if args.log_to_file else None
    setup_logging(log_level=log_level, log_to_file=args.log_to_file, log_dir=log_dir)

    logging.debug(f'Configuration:\n{args}')
    
    # Define paths
    root_dir = Path(args.source)
    output_dir = Path(args.output)
    processed_log = Path(args.processed_log)
    
    # Handle summary and failure options separately
    if args.summary or args.show_failures:
        # Load processed log data for summary and failure reporting
        processed_data = load_processed_data(processed_log)
        if args.summary:
            total_processed = len(processed_data)
            print(f"Total files processed: {total_processed}")
    
        if args.show_failures:
            failures = [file for file, data in processed_data.items() if data["status"] == "failure"]
            if failures:
                print("Files that failed processing:")
                for file in failures:
                    print(f"- {file}")
            else:
                print("No files failed processing.")
        return

    print("Starting MP3 file processing...")

    # Locate MP3 files
    mp3_files = locate_mp3_files(root_dir)
    if not mp3_files:
        print("No MP3 files found to process.")
        return

    # Flag for quit-after-current behavior
    quit_after_current = False

    try:
        # Process each located MP3 file with specified chunk length
        for file_path in mp3_files:
            try:
                # Process a single MP3 file
                process_mp3_files([file_path], output_dir=output_dir, log_file=processed_log, chunk_length=args.chunk_length * 1000)
                logging.debug(f"Finished processing file: {file_path.name}")

                # Break loop if user has requested to quit after the current track
                if quit_after_current:
                    print("Quitting after current track as requested.")
                    break

            except KeyboardInterrupt:
                if quit_after_current:
                    print("Immediate quit requested. Exiting now!")
                    break
                else:
                    quit_after_current = True
                    print(f"\n\n{'#'*30}\n     Ctrl+C Caught\n{'#'*30}\n")
                    print("Press Ctrl+C again to quit immediately, or wait for current track to complete and then exit.\n")
                    time.sleep(0.1)  # Small pause to avoid double-triggering

    except KeyboardInterrupt:
        logging.info("Processing interrupted. Exiting...")

    print("MP3 file processing completed.")




if __name__ == "__main__":
    main()
