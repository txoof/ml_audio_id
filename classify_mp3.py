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
import re

import pygame
# from mutagen.mp3 import MP3
# from mutagen.id3 import ID3, TIT2


logger = logging.getLogger(__name__)

# +
import pygame
import logging
import sys
import json
import argparse
from pathlib import Path
import re

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

def play_audio(file_path):
    """Initialize and play an MP3 file."""
    pygame.mixer.init()
    pygame.mixer.music.load(str(file_path))
    pygame.mixer.music.play()
    logging.debug(f"Playing audio file: {file_path}")

def stop_audio():
    """Stop the currently playing audio."""
    pygame.mixer.music.stop()
    logging.debug("Audio stopped.")

def get_formatted_title(file_path):
    """Format the title by removing underscores from the filename."""
    return file_path.stem.replace("_", " ")

def get_chunk_number(file_path):
    """Extract the chunk number from the filename."""
    match = re.search(r'chunk_(\d+)', file_path.stem)
    return match.group(1) if match else "Unknown"

def render_wrapped_text(surface, text, font, color, x, y, max_width):
    """Render wrapped text onto a pygame surface."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word would exceed max width
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "

    # Append the last line
    lines.append(current_line)

    # Render each line and blit it onto the surface
    for i, line in enumerate(lines):
        line_surface = font.render(line, True, color)
        surface.blit(line_surface, (x, y + i * font.get_height()))


# -

def parse_arguments():
    """Parse command-line arguments for the MP3 player."""
    parser = argparse.ArgumentParser(description="MP3 Player with Classification")
    parser.add_argument("mp3_directory", type=str, help="Directory containing MP3 files to classify")
    parser.add_argument("--record_file", type=str, default="track_classifications.json", help="File to store classification records (JSON format)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (use -vv for DEBUG)")
    parser.add_argument("--log_to_file", action="store_true", help="Enable logging to a file in the logs directory")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level based on verbosity
    log_level = logging.WARNING  # Default level
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    # Set up logging with the specified verbosity and optional file logging
    setup_logging(log_level=log_level, log_to_file=args.log_to_file)

    # Set MP3 directory and record file
    MP3_DIRECTORY = Path(args.mp3_directory)
    RECORD_FILE = Path(args.record_file)

    # Locate MP3 files in the specified directory
    mp3_files = locate_mp3_files(MP3_DIRECTORY)

    # Load existing classifications or initialize an empty dictionary
    if RECORD_FILE.exists():
        with RECORD_FILE.open("r") as file:
            classifications = json.load(file)
        logging.info(f"Loaded existing classifications from '{RECORD_FILE}'.")
    else:
        classifications = {}
        logging.info(f"No existing classification file found. Starting fresh.")

    pygame.init()
    screen = pygame.display.set_mode((800, 400), pygame.RESIZABLE)  # Resizable, larger display window
    pygame.display.set_caption("MP3 Player")
    clock = pygame.time.Clock()
    paused = False

    # Set up fonts for displaying text
    font = pygame.font.Font(None, 24)  # Reduced font size for track name
    control_font = pygame.font.Font(None, 20)

    # Define control text with added "Shift + Left Arrow = Previous Track"
    control_text = "Controls: Space = Play/Pause | Right Arrow = Fast Forward | Left Arrow = Rewind | Shift + Left Arrow = Previous Track | Q = Quit"

    # Initialize track index
    current_track_index = 0

    try:
        while current_track_index < len(mp3_files):
            file_path = mp3_files[current_track_index]

            # Skip previously classified tracks
            if file_path.stem in classifications:
                logging.info(f"Skipping previously classified track: {file_path.stem}")
                current_track_index += 1
                continue

            # Format the title from the filename and extract chunk number
            title = get_formatted_title(file_path)
            chunk_number = get_chunk_number(file_path)
            chunk_text = f"Chunk: {chunk_number}"
            wrapped_title = title  # Title to be wrapped and displayed after "Chunk: NN"

            play_audio(file_path)

            while pygame.mixer.music.get_busy() or paused:
                # Clear the screen
                screen.fill((0, 0, 0))

                # Render and display chunk number
                chunk_surface = font.render(chunk_text, True, (255, 255, 255))
                screen.blit(chunk_surface, (20, 80))

                # Render and display wrapped title below chunk number
                render_wrapped_text(screen, wrapped_title, font, (255, 255, 255), 20, 120, screen.get_width() - 40)

                # Render and display controls at the bottom
                render_wrapped_text(screen, control_text, control_font, (200, 200, 200), 20, screen.get_height() - 60, screen.get_width() - 40)

                # Update the display
                pygame.display.flip()

                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_audio()
                        pygame.quit()
                        sys.exit()  # Use sys.exit to close cleanly
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # Toggle pause and resume
                            if paused:
                                pygame.mixer.music.unpause()
                            else:
                                pygame.mixer.music.pause()
                            paused = not paused
                        elif event.key == pygame.K_RIGHT:
                            # Fast forward by 10 seconds
                            current_pos = pygame.mixer.music.get_pos() / 1000.0 + 10
                            pygame.mixer.music.set_pos(current_pos)
                        elif event.key == pygame.K_LEFT and (event.mod & pygame.KMOD_SHIFT):
                            # Go back to previous track
                            if current_track_index > 0:
                                current_track_index -= 1
                                logging.info(f"Going back to previous track: {mp3_files[current_track_index].stem}")
                                stop_audio()
                                break  # Exit the loop to go to the previous track
                        elif event.key == pygame.K_LEFT:
                            # Rewind by 10 seconds
                            current_pos = max(0, pygame.mixer.music.get_pos() / 1000.0 - 10)
                            pygame.mixer.music.set_pos(current_pos)
                        elif event.key == pygame.K_q:
                            # Quit the program
                            stop_audio()
                            pygame.quit()
                            sys.exit()  # Use sys.exit to close cleanly
                    elif event.type == pygame.VIDEORESIZE:
                        # Handle window resizing
                        screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                # Small delay to prevent excessive CPU usage
                clock.tick(10)
            
            # Stop the music if it hasn't been stopped already
            stop_audio()

            # Prompt user for classification with Quit option
            prompt_text = "Choose classification: D = Dialogue, M = Music, B = Both, N = None, S = Skip, Q = Quit"
            render_wrapped_text(screen, prompt_text, font, (255, 255, 255), 20, 200, screen.get_width() - 40)
            pygame.display.flip()

            classification = None
            waiting_for_choice = True
            while waiting_for_choice:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            classification = "Dialogue"
                            waiting_for_choice = False
                        elif event.key == pygame.K_m:
                            classification = "Music"
                            waiting_for_choice = False
                        elif event.key == pygame.K_b:
                            classification = "Both"
                            waiting_for_choice = False
                        elif event.key == pygame.K_n:
                            classification = "None"
                            waiting_for_choice = False
                        elif event.key == pygame.K_s:
                            classification = "Skip"
                            waiting_for_choice = False
                        elif event.key == pygame.K_q:
                            # Quit option during classification
                            logging.info("User chose to quit during classification.")
                            pygame.quit()
                            sys.exit()  # Use sys.exit to exit cleanly

            # Record classification if not skipped
            if classification != "Skip":
                classifications[file_path.stem] = classification
                with RECORD_FILE.open("w") as file:
                    json.dump(classifications, file, indent=2)
                logging.info(f"Recorded classification for {file_path.stem}: {classification}")

            # Move to the next track
            current_track_index += 1

        logging.info("All tracks reviewed.")
    finally:
        # Ensure pygame exits cleanly on errors or completion
        pygame.quit()
        sys.exit()  # Ensure clean exit if not already exited


# +
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level based on verbosity
    log_level = logging.WARNING  # Default level
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    # Set up logging with the specified verbosity and optional file logging
    setup_logging(log_level=log_level, log_to_file=args.log_to_file)

    # Set MP3 directory and record file
    MP3_DIRECTORY = Path(args.mp3_directory)
    RECORD_FILE = Path(args.record_file)

    # Locate MP3 files in the specified directory
    mp3_files = locate_mp3_files(MP3_DIRECTORY)

    # Load existing classifications or initialize an empty dictionary
    if RECORD_FILE.exists():
        with RECORD_FILE.open("r") as file:
            classifications = json.load(file)
        logging.info(f"Loaded existing classifications from '{RECORD_FILE}'.")
    else:
        classifications = {}
        logging.info(f"No existing classification file found. Starting fresh.")

    # Calculate initial statistics
    total_mp3s = len(mp3_files)
    categorized_mp3s = len(classifications)
    remaining_mp3s = total_mp3s - categorized_mp3s

    pygame.init()
    screen = pygame.display.set_mode((800, 400), pygame.RESIZABLE)  # Resizable, larger display window
    pygame.display.set_caption("MP3 Player")
    clock = pygame.time.Clock()
    paused = False

    # Set up fonts for displaying text
    font = pygame.font.Font(None, 24)  # Reduced font size for track name
    control_font = pygame.font.Font(None, 20)

    # Define control text with "Down Arrow = Previous Track"
    control_text = "Controls: Space = Play/Pause | Right Arrow = Fast Forward | Left Arrow = Rewind | Down Arrow = Previous Track | Q = Quit"

    # Initialize track index
    current_track_index = 0
    track_navigation = False

    try:
        while current_track_index < len(mp3_files):
            file_path = mp3_files[current_track_index]

            # Skip previously classified tracks
            if file_path.stem in classifications and not track_navigation:
                logging.info(f"Skipping previously classified track: {file_path.stem}")
                current_track_index += 1
                continue

            # Update remaining count if we’re playing a new track
            if not track_navigation:
                remaining_mp3s = total_mp3s - (categorized_mp3s + current_track_index)

            # Format the title from the filename and extract chunk number
            title = get_formatted_title(file_path)
            chunk_number = get_chunk_number(file_path)
            chunk_text = f"Chunk: {chunk_number}"
            wrapped_title = title  # Title to be wrapped and displayed after "Chunk: NN"

            play_audio(file_path)

            while pygame.mixer.music.get_busy() or paused:
                # Clear the screen
                screen.fill((0, 0, 0))

                # Display track statistics above track details
                stats_text = f"Total MP3s: {total_mp3s} | Categorized: {categorized_mp3s} | Remaining: {remaining_mp3s}"
                stats_surface = font.render(stats_text, True, (255, 255, 255))
                screen.blit(stats_surface, (20, 20))

                # Render and display chunk number
                chunk_surface = font.render(chunk_text, True, (255, 255, 255))
                screen.blit(chunk_surface, (20, 80))

                # Render and display wrapped title below chunk number
                render_wrapped_text(screen, wrapped_title, font, (255, 255, 255), 20, 120, screen.get_width() - 40)

                # Render and display controls at the bottom
                render_wrapped_text(screen, control_text, control_font, (200, 200, 200), 20, screen.get_height() - 60, screen.get_width() - 40)

                # Update the display
                pygame.display.flip()

                track_navigation = False
                
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_audio()
                        pygame.quit()
                        sys.exit()  # Use sys.exit to close cleanly
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # Toggle pause and resume
                            if paused:
                                pygame.mixer.music.unpause()
                            else:
                                pygame.mixer.music.pause()
                            paused = not paused
                        elif event.key == pygame.K_RIGHT:
                            # Pause, fast forward by 2 seconds, then resume
                            paused = True
                            pygame.mixer.music.pause()
                            current_pos = pygame.mixer.music.get_pos() / 1000.0 + 3
                            pygame.mixer.music.set_pos(current_pos)
                            pygame.mixer.music.unpause()
                            paused = False
                        # elif event.key == pygame.K_RIGHT:
                        #     # Fast forward by 2 seconds
                        #     current_pos = pygame.mixer.music.get_pos() / 1000.0 + 2
                        #     pygame.mixer.music.set_pos(current_pos)
                        elif event.key == pygame.K_DOWN:
                            # Go back to previous track with Down Arrow
                            if current_track_index > 0:
                                current_track_index -= 1
                                logging.info(f"Going back to previous track: {mp3_files[current_track_index].stem}")
                                stop_audio()
                                track_navigation = True
                                break  # Exit the loop to go to the previous track
                        elif event.key == pygame.K_LEFT:
                            # Rewind by 10 seconds
                            current_pos = max(0, pygame.mixer.music.get_pos() / 1000.0 - 10)
                            pygame.mixer.music.set_pos(current_pos)
                        elif event.key == pygame.K_q:
                            # Quit the program
                            stop_audio()
                            pygame.quit()
                            sys.exit()  # Use sys.exit to close cleanly
                    elif event.type == pygame.VIDEORESIZE:
                        # Handle window resizing
                        screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                # Small delay to prevent excessive CPU usage
                clock.tick(10)
            if track_navigation:
                continue
            
            # Stop the music if it hasn't been stopped already
            stop_audio()

            # Prompt user for classification with Quit option
            prompt_text = "Choose classification: D = Dialogue, M = Music, B = Both, N = None, S = Skip, Q = Quit"
            render_wrapped_text(screen, prompt_text, font, (255, 255, 255), 20, 200, screen.get_width() - 40)
            pygame.display.flip()

            classification = None
            waiting_for_choice = True
            while waiting_for_choice:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            classification = "Dialogue"
                            waiting_for_choice = False
                        elif event.key == pygame.K_m:
                            classification = "Music"
                            waiting_for_choice = False
                        elif event.key == pygame.K_b:
                            classification = "Both"
                            waiting_for_choice = False
                        elif event.key == pygame.K_n:
                            classification = "None"
                            waiting_for_choice = False
                        elif event.key == pygame.K_s:
                            classification = "Skip"
                            waiting_for_choice = False
                        elif event.key == pygame.K_q:
                            # Quit option during classification
                            logging.info("User chose to quit during classification.")
                            pygame.quit()
                            sys.exit()  # Use sys.exit to exit cleanly

            # Record classification if not skipped
            if classification != "Skip":
                classifications[file_path.stem] = classification
                with RECORD_FILE.open("w") as file:
                    json.dump(classifications, file, indent=2)
                logging.info(f"Recorded classification for {file_path.stem}: {classification}")
                categorized_mp3s += 1  # Update categorized count

            # Move to the next track
            current_track_index += 1

        logging.info("All tracks reviewed.")
    finally:
        # Ensure pygame exits cleanly on errors or completion
        pygame.quit()
        sys.exit()  # Ensure clean exit if not already exited

if __name__ == "__main__":
    main()

