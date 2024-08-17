
# AudioInsight: Audio File Analysis and Insights

**AudioInsight** is a Python tool designed for audio analysis, offering functionalities such as BPM detection, key detection, and integration with a GUI for easy file selection and analysis. It processes audio files, extracts metadata, and provides visual progress using the Tkinter framework.

## Features

- **Audio Metadata Extraction**: Retrieves information such as bitrate and audio tags (supports MP3 and FLAC).
- **BPM Detection**: Detects beats per minute (BPM) using the `librosa` library.
- **Key Detection**: Determines the key of an audio file by analyzing chroma features.
- **Graphical Interface**: Simple GUI for selecting audio files or directories for analysis.
- **Multithreading**: Uses a thread pool for efficient concurrent processing of audio files.
- **Error Logging**: Errors are logged to `audioinsight.log` for debugging.

## Prerequisites

- Python 3.x
- Libraries: Install the following dependencies using pip:

```bash
pip install pydub eyed3 pandas numpy librosa pyloudnorm music21 tk mutagen pillow
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/SynBioExplorer/AudioInsight.git
cd AudioInsight
```

2. Install the required Python libraries (as listed in the prerequisites).

3. Run the script:

```bash
python AudioInsight.py
```

4. The graphical interface will open, allowing you to select the directory containing audio files for analysis.

### Example GUI Interface

Here’s how the graphical interface looks when you run the program:

![GUI Example](/mnt/data/AudioInsight_GUI.png)

### Output File Structure

After processing, the script will generate output in a structured format like this:

| file_name                               | file_type | frame_rate | channels | bit_depth | bit_rate | loudness_lufs | bpm | key     |
|-----------------------------------------|-----------|------------|----------|-----------|----------|---------------|-----|---------|
| Fyex - Around The World.mp3             | .mp3      | 44100      | 2        | 16        | 320      | -10.26661536  | 144 | F minor |
| because_you_move_me.mp3                 | .mp3      | 48000      | 2        | 16        | 320      | -11.69597492  | 123 | F major |
| Bon Entendeur - Nan Ye Li Kan.mp3       | .mp3      | 44100      | 2        | 16        | 320      | -13.57232696  | 123 | D minor |
| Bennett.flac                            | .flac     | 44100      | 2        | 16        | 895363   | -10.19902104  | 136 | D major |

### Example Spectrogram

Here’s an example of the resulting spectrogram produced for an audio file:

![Spectrogram Example](/mnt/data/AudioInsight_Spectrogram.png)

### Directory Structure Example

```
/AudioInsight
    ├── AudioInsight.py    # Main script
    ├── audioinsight.log   # Log file
    └── README.md          # This README file
```

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2024 Felix Meier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
