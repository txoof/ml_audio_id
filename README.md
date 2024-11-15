# Project Title: ML_Audio_ID - Identifying Music in Audio Tracks Using Machine Learning <!-- omit from toc -->

Use Machine Learning to positively identify if an audio clip includes music.

- [Project Details](#project-details)
- [Project Journal and Report](#project-journal-and-report)
- [Scripts Generated for this Project](#scripts-generated-for-this-project)
  - [Pre-Processing](#pre-processing)
  - [Utilities](#utilities)

## Project Details

### Project Scope <!-- omit from toc -->
This project will focus on using existing machine learning algorithms to develop a system that accurately determines whether a short audio track contains music. The solution aims to lay the foundation for a larger project involving the identification of musical pieces used in broadcast media, specifically on National Public Radio (NPR).

### Problem Statement <!-- omit from toc -->

National Public Radio frequently uses interesting music to close out news and editorial segments. Unfortunately, they do not consistently publish the title or artist of these tracks, which leaves the listening community without an easy way to discover the music they find intriguing. Correctly identifying tracks that contain outro music is a crucial first step in a chain that ultimately allows listeners to discover the music featured in these segments. Given that API-based music identification services charge per job, it's important to avoid unnecessary processing of tracks that do not contain music. This project aims to build a system that ensures only relevant audio tracks (i.e., those containing music) are sent for further identification, thereby saving resources and reducing costs.

### Project Goals <!-- omit from toc -->

- Develop a reliable system that accurately identifies audio tracks that contain music.
- Explore the effectiveness of different machine learning algorithms in solving this problem.
- Gain a deeper understanding of audio data processing and classification using machine learning.

### Tools and Resources <!-- omit from toc -->

The project will utilize existing Python-based machine learning libraries to determine their effectiveness and applicability for this task. A training corpus of audio tracks, consisting of segments with and without music, is being compiled and will be used for training and evaluation.

### Project Timeline <!-- omit from toc -->

The project duration is two weeks, broken down as follows:

- **Week 1**:
  - **Days 1-2**: Define the problem in greater detail and compile and label the training dataset.
  - **Days 3-5**: Explore existing ML algorithms and develop an initial prototype of the audio classification system.
- **Week 2**:
  - **Days 6-8**: Evaluate the performance of different algorithms, optimize the solution, and finalize the model.
  - **Days 9-10**: Prepare the final report and documentation, and compile the codebase for submission.

### Deliverables <!-- omit from toc -->

- A short report detailing the machine learning algorithms explored, the evaluation of their effectiveness for the task, and a summary of findings.
- A Git repository containing the project code, including Jupyter Notebooks, scripts, and supporting Markdown documents to explain the workflow, methodologies used, and the results obtained.

### Significance <!-- omit from toc -->

This project supports personal growth by enhancing skills related to AI, data science, and self-directed research. It offers hands-on experience with machine learning algorithms and showcases the development of a solution with direct connection to the program.

## Project Journal and Report

Daily details of progress and steps taken can be found in the [Project_Journal.md](./Project_Journal.md). 
A summary of learning can be found in the [Project Report](./Challenge_Project_Report.md)

## Scripts Generated for this Project

### Pre-Processing

These scripts are used for pre-processing the audio files in preparation for classification

#### [`chunk.py`](./chunk_mp3.py)<!-- omit from toc -->

Chunk source data into 10-second mp3 clips starting from the last 10 seconds. The last 10 seconds are the most significant for classification. The remaining clips can be used as additional training data to enrich the set if needed. Most of the remaining clips contain dialogue.

Processed files are recorded in `./processed_files.json` by default to prevent re-chunking new files.

```BASH
usage: chunk_mp3.py [-h] [-v] [-l LOG_PATH] [--log_to_file] [-s SOURCE] [-o OUTPUT] [-p PROCESSED_LOG]
                    [--chunk_length CHUNK_LENGTH] [-y] [-f]

Process MP3 files into specified-length chunks from end to start.

options:
  -h, --help            show this help message and exit
  -v                    Increase logging verbosity (e.g., -v, -vv, -vvv)
  -l LOG_PATH, --log_path LOG_PATH
                        Directory for log file output if logging to file
  --log_to_file         Log output to a file instead of screen (default: screen)
  -s SOURCE, --source SOURCE
                        Root directory to search for MP3 files
  -o OUTPUT, --output OUTPUT
                        Directory to save output chunks
  -p PROCESSED_LOG, --processed_log PROCESSED_LOG
                        JSON file to log processed files
  --chunk_length CHUNK_LENGTH
                        Length of each audio chunk in seconds (default: 10)
  -y, --summary         Show a summary of how many files were processed
  -f, --show_failures   Show a list of all files that failed processing
```

#### ['classify_mp3.py](./classify_mp3.py)

Launch an interface for manually labeling mp3 tracks into three categories:

- Dialogue (containing primarily radio dialogue)
- Music (containing primarily music)
- Both (containing both dialogue and music)

Classified files are stored in`./track_classifications.json` by default. This can be used as "ground truth" for labeling datasets.

```BASH
usage: classify_mp3.py [-h] [-l LOG_PATH] [--record_file RECORD_FILE] [-v] [--log_to_file] mp3_directory

MP3 Player with Classification

positional arguments:
  mp3_directory         Directory containing MP3 files to classify

options:
  -h, --help            show this help message and exit
  -l LOG_PATH, --log_path LOG_PATH
                        Directory for log file output if logging to file
  --record_file RECORD_FILE
                        File to store classification records (JSON format)
  -v, --verbose         Increase verbosity level (use -vv for DEBUG)
  --log_to_file         Enable logging to a file in the logs director
```


### [Utilities](./utilities/)

#### [`fetch_data`](./utilities/fetch_data)<!-- omit from toc -->

Bash script to non-interactively rsync untagged mp3 data from local server to local machine for further processing

```BASH
Usage: ./fetch_data [HOST] [PORT] [USER] [REMOTE_SOURCE] [LOCAL_DEST]
Default HOST: 192.168.1.9
Default PORT: 22
Default USER: media
Default REMOTE_SOURCE: ~/TRAINING_DATA/
Default LOCAL_DEST: /Users/aaronciuffo/Documents/src/ml_audio_id/TRAINING_DATA/RAW
```

#### [`init_environment.sh`](./utilities/init_environment.sh)<!-- omit from toc -->

Bash script to build virtual environment and install required python libraries. This will install all requirements found in `requirements.txt` and `utiltities/requirements-jupyter_devel.txt`

```BASH
  Setup the virtual environment for this project

  usage:
  $ ./init_environment.sh [option]

  options:
  -c: create the virtual environment
  -j: create the virtual environment and add jupyter kernel for development
  -h: This help screen
  -p: purge the virtual environment and clean jupyter kernelspecs
  --info: virtual environment information
```
