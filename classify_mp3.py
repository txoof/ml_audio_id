#! /usr/bin/env python
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

import logging
from pathlib import Path
import os
import sys
import argparse
import time
import json
import logging
import argparse

import pygame
# from mutagen.mp3 import MP3
# from mutagen.id3 import ID3, TIT2


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


def locate_mp3_files(root_dir, glob="*.mp3"):
    """Recursively locate all MP3 files within a given directory path."""
    mp3_files = []
    root_path = Path(root_dir)
    logging.info(f"Searching '{root_dir}' for '{glob}' files.")
    if root_path.exists() and root_path.is_dir():
        mp3_files = list(root_path.rglob(glob))
        logging.info(f"Found {len(mp3_files)} MP3 files in '{root_dir}'.")
    else:
        logging.warning(f"Directory '{root_dir}' does not exist or is not accessible.")
    return mp3_files


def parse_arguments():
    """Parse command-line arguments for the MP3 player."""
    parser = argparse.ArgumentParser(description="MP3 Player with Classification")
    parser.add_argument("mp3_directory", type=str, help="Directory containing MP3 files to classify")
    parser.add_argument("-l", "--log_path", type=str, help="Directory for log file output if logging to file")
    parser.add_argument("--record_file", type=str, default="track_classifications.json", help="File to store classification records (JSON format)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (use -vv for DEBUG)")
    parser.add_argument("--log_to_file", action="store_true", help="Enable logging to a file in the logs directory")
    return parser.parse_args()


def play_mp3(file_path):
    # Load and play an mp3 file
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


# +
def save_classification(mp3, current_tag, classifications, record_file):
    if current_tag != "None" and mp3.name not in classifications:
        classifications[mp3.name] = current_tag
        with record_file.open("w") as file:
            json.dump(classifications, file, indent=4)
        logging.info(f"Classification for '{mp3.name}' saved as '{current_tag}'.")

def main():
    # Initialize pygame
    pygame.init()

    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level based on verbosity count (-v)
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]

    # Set up logging with the specified verbosity and optional file logging
    setup_logging(log_level=log_level, log_to_file=args.log_to_file)

    # Set MP3 directory and record file
    MP3_DIRECTORY = Path(args.mp3_directory)
    RECORD_FILE = Path(args.record_file)

    # Load existing classifications or initialize an empty dictionary
    if RECORD_FILE.exists():
        with RECORD_FILE.open("r") as file:
            classifications = json.load(file)
        logging.info(f"Loaded existing classifications from '{RECORD_FILE}'.")
    else:
        classifications = {}
        logging.info(f"No existing classification file found. Starting fresh.")

    # Locate MP3 files in the specified directory and prune out any files that appear in the record file
    mp3_files = [mp3 for mp3 in locate_mp3_files(MP3_DIRECTORY) if mp3.name not in classifications]
    
    # Set up pygame window
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('MP3 Classification')
    font = pygame.font.Font(None, 24)
    progress_font = pygame.font.Font(None, 36)
    control_font = pygame.font.Font(None, 28)  # Increase font size for better readability
    tag_font = pygame.font.Font(None, 28)  # Font for audio tagging controls

    current_tag = "None"
    tagged_count = 0

    # Main loop
    running = True
    selected_index = 0
    while running:
        mp3 = mp3_files[selected_index]
        current_tag = classifications.get(mp3.name, "None")
        
        # Display the total number of items in the mp3_files list
        screen.fill((30, 30, 30))
        total_items_text = f"Tagged: {tagged_count}/{len(mp3_files)}"
        total_items_surface = font.render(total_items_text, True, (255, 255, 255))
        screen.blit(total_items_surface, (50, 10))

        # Display the selected MP3 file name
        name = ' '.join(mp3.name.split('_'))
        text = font.render(name, True, (255, 255, 255))
        screen.blit(text, (50, 50))
        pygame.display.flip()

        # Play the selected MP3 file
        play_mp3(mp3)
        track_length = pygame.mixer.Sound(mp3).get_length()
        start_time = time.time()

        # Wait for user to press a key
        waiting_for_key = True
        while waiting_for_key:
            # Update the track progress display
            elapsed_time = time.time() - start_time
            remaining_time = max(0, track_length - elapsed_time)
            progress_text = f"Total Length: {int(track_length)}s / Remaining: {int(remaining_time)}s"
            progress = elapsed_time / track_length if track_length > 0 else 0

            # Draw progress text and progress bar
            screen.fill((30, 30, 30), (50, 100, 900, 300))  # Clear previous progress and controls
            progress_text_surface = progress_font.render(progress_text, True, (255, 255, 255))
            screen.blit(progress_text_surface, (50, 100))
            pygame.draw.rect(screen, (0, 255, 0), (50, 150, int(900 * progress), 20))

            # Draw current tag
            pygame.draw.rect(screen, (100, 100, 100), (45, 230, 910, 40))  # Gray box for current tag
            current_tag_surface = control_font.render(f"Current Tag: {current_tag}", True, (255, 255, 255))
            screen.blit(current_tag_surface, (50, 240))

            # Draw tagging options
            tag_text = "Audio Tag: [D]ialogue | [M]usic | [B]oth | [N]one"
            tag_text_surface = tag_font.render(tag_text, True, (255, 255, 255))
            screen.blit(tag_text_surface, (50, 280))

            # Draw control options
            control_text = "Controls: [UP] Prev | [DOWN] Next | [LEFT] Rew | [RIGHT] FF | [Q]uit"
            control_text_surface = control_font.render(control_text, True, (255, 255, 255))
            screen.blit(control_text_surface, (50, 320))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if current_tag != "None" and mp3.name not in classifications:
                        save_classification(mp3, current_tag, classifications, RECORD_FILE)
                        tagged_count += 1
                        tagged_count += 1
                    running = False
                    waiting_for_key = False
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_index = (selected_index - 1) % len(mp3_files)
                        waiting_for_key = False
                    elif event.key == pygame.K_DOWN:
                        save_classification(mp3, current_tag, classifications, RECORD_FILE)
                        if current_tag != "None" and mp3.name not in classifications:
                            tagged_count += 1
                        selected_index = (selected_index + 1) % len(mp3_files)
                        waiting_for_key = False
                    # Handle fast-forward action only if the music is playing
                    if event.key == pygame.K_RIGHT:
                        if pygame.mixer.music.get_busy():
                            new_pos = pygame.mixer.music.get_pos() / 1000 + 1
                            if new_pos < track_length:
                                pygame.mixer.music.set_pos(new_pos)
                                start_time -= 2
                            pygame.mixer.music.set_pos(new_pos)
                            start_time -= 2
                    # Handle rewind action only if the music is playing
                    if event.key == pygame.K_LEFT:
                        if pygame.mixer.music.get_busy():
                            new_pos = max(0, pygame.mixer.music.get_pos() / 1000 - 3)
                            pygame.mixer.music.set_pos(new_pos)
                            start_time += 2
                        start_time += 2
                    elif event.key == pygame.K_RETURN:
                        waiting_for_key = False
                    elif event.key == pygame.K_q:
                        save_classification(mp3, current_tag, classifications, RECORD_FILE)
                        if current_tag != "None" and mp3.name not in classifications:
                            tagged_count += 1
                        running = False
                        waiting_for_key = False
                    elif event.key == pygame.K_d:
                        current_tag = "Dialogue"
                    elif event.key == pygame.K_m:
                        current_tag = "Music"
                    elif event.key == pygame.K_b:
                        current_tag = "Both"
                    elif event.key == pygame.K_n:
                        current_tag = "None"

    # Quit pygame
    pygame.quit()
    


# -

if __name__ == "__main__":
    main()


