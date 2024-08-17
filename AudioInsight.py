#!/usr/local/bin/python3

import os
import sys
import logging
import eyed3
import pandas as pd
import numpy as np
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyloudnorm as pyln
import functools
import librosa
from music21 import stream, note, chord
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from mutagen.flac import FLAC
import threading
from PIL import Image, ImageTk

# Setup logging
logging.basicConfig(filename='audioinsight.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_audio(file_path):
    try:
        audio_segment = AudioSegment.from_file(file_path)
        if file_path.endswith('.flac'):
            audio_tag = FLAC(file_path)
            bit_rate = audio_tag.info.bitrate if audio_tag.info else "Unknown"
        else:
            audio_tag = eyed3.load(file_path)
            bit_rate = audio_tag.info.bit_rate[1] if audio_tag.info and audio_tag.info.bit_rate else "Unknown"
        return audio_segment, audio_tag, bit_rate
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None, None, "Unknown"

def detect_bpm(file_path):
    try:
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return round(tempo)
    except Exception as e:
        logging.error(f"Error detecting BPM for {file_path}: {e}")
        return None

def detect_key(file_path):
    try:
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        s = stream.Score(id='mainScore')
        
        for i in range(chroma.shape[1]):
            pitch_classes = np.where(chroma[:, i] > 0.1)[0]
            if len(pitch_classes) > 0:
                notes = [note.Note(pc + 60) for pc in pitch_classes]
                c = chord.Chord(notes)
                s.append(c)

        key = s.analyze('key')
        return key.tonic.name + ' ' + key.mode
    except Exception as e:
        logging.error(f"Error detecting key for {file_path}: {e}")
        return None

def get_audio_details(audio_segment, audio_tag, bit_rate):
    try:
        properties = {
            'frame_rate': audio_segment.frame_rate,
            'channels': audio_segment.channels,
            'bit_depth': audio_segment.sample_width * 8,
            'bit_rate': bit_rate
        }
        return properties
    except Exception as e:
        logging.error(f"Error getting audio details: {e}")
        return None

def measure_loudness(audio_segment):
    try:
        data = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            data = data.reshape((-1, 2))
            data = data.mean(axis=1)
        data = data.astype(np.float32) / (2**15)
        meter = pyln.Meter(audio_segment.frame_rate)
        return meter.integrated_loudness(data)
    except Exception as e:
        logging.error(f"Error measuring loudness: {e}")
        return None

def plot_spectrogram(audio_segment, file_name, spectrogram_directory):
    try:
        data = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            data = data.reshape((-1, 2))
            data = data.mean(axis=1)
        
        data = data.astype(np.float32)
        data /= np.max(np.abs(data))

        D = librosa.stft(data, n_fft=2048, hop_length=512)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        sr = audio_segment.frame_rate
        times = librosa.times_like(D, sr=sr, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048) / 1000  # in kHz

        fig, ax = plt.subplots(figsize=(20, 8))
        img = ax.imshow(D_db, aspect='auto', origin='lower', cmap='inferno', 
                        extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        
        ax.set_title('Spectrogram of ' + file_name)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (kHz)')
        
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Intensity (dB)')
        
        ensure_dir(spectrogram_directory)
        fig.savefig(os.path.join(spectrogram_directory, file_name + '.png'))
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error plotting spectrogram: {e}")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_file(file_path, spectrogram_directory):
    try:
        audio_segment, audio_tag, bit_rate = load_audio(file_path)
        if audio_segment is None or audio_tag is None:
            return None
        audio_properties = get_audio_details(audio_segment, audio_tag, bit_rate)
        audio_properties['loudness_lufs'] = measure_loudness(audio_segment)
        audio_properties['file_name'] = os.path.basename(file_path)
        audio_properties['bpm'] = detect_bpm(file_path)
        audio_properties['key'] = detect_key(file_path)
        audio_properties['file_type'] = os.path.splitext(file_path)[1]
        plot_spectrogram(audio_segment, os.path.splitext(os.path.basename(file_path))[0], spectrogram_directory)
        return audio_properties
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def save_dataframe_to_excel(df, output_dir):
    try:
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'output.xlsx')
        df.to_excel(output_file, index=False)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving dataframe to Excel: {e}")


def run_audio_analysis(input_dir, progress):
    spectrogram_directory = os.path.join(input_dir, 'spectrograms')
    ensure_dir(spectrogram_directory)
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames if f.endswith(('.mp3', '.wav', '.flac'))]

    progress['maximum'] = len(all_files)
    progress['value'] = 0

    audio_data = []

    def update_progress(future):
        progress['value'] += 1
        root.update_idletasks()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_file, file_path, spectrogram_directory): file_path for file_path in all_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                audio_data.append(result)
            update_progress(future)

    if audio_data:
        df = pd.DataFrame(audio_data)
        column_order = ['file_name', 'file_type'] + [col for col in df.columns if col not in ['file_name', 'file_type']]
        df = df.reindex(columns=column_order)
        save_dataframe_to_excel(df, input_dir)
        threading.Thread(target=lambda: messagebox.showinfo("Success", "Audio analysis completed successfully!")).start()
    else:
        threading.Thread(target=lambda: messagebox.showinfo("Error", "No audio data was processed.")).start()
        logging.info("No audio data was processed.")


def start_analysis(input_dir, progress, run_btn):
    run_btn.config(text="Processing...", state=tk.DISABLED)
    def run_and_enable_button():
        run_audio_analysis(input_dir, progress)
        run_btn.config(text="Run Analysis", state=tk.NORMAL)
    threading.Thread(target=run_and_enable_button).start()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main_app():
    global root
    root = tk.Tk()
    root.title("Audio Insight")

    try:
        background_image_path = resource_path("background.jpg")

        # Load and resize the background image
        background_image = Image.open(background_image_path)
        window_width, window_height = 800, 600  # Example window size, adjust as needed
        background_image = background_image.resize((window_width, window_height), Image.LANCZOS)
        background_photo = ImageTk.PhotoImage(background_image)

        # Create canvas to hold the background image
        canvas = tk.Canvas(root, width=window_width, height=window_height)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=background_photo, anchor="nw")

        # Define styles
        frame = tk.Frame(root, bg="white", padx=20, pady=20)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        input_dir_label = tk.Label(frame, text="Select Input Directory:", bg="white", fg="#000000", font=("Helvetica", 12))
        input_dir_label.pack(pady=(0, 5))
        input_dir_entry = tk.Entry(frame, width=50)
        input_dir_entry.pack(pady=5)
        input_dir_btn = tk.Button(frame, text="Browse", command=lambda: input_dir_entry.insert(0, filedialog.askdirectory()), bg="#000000", fg="#000000", font=("Helvetica", 12))
        input_dir_btn.pack(pady=5)

        progress_label = tk.Label(frame, text="Progress:", bg="white", fg="#000000", font=("Helvetica", 12))
        progress_label.pack(pady=(15, 5))
        progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        progress.pack(pady=10)

        run_btn = tk.Button(frame, text="Run Analysis", command=lambda: start_analysis(input_dir_entry.get(), progress, run_btn), bg="#000000", fg="#000000", font=("Helvetica", 12))
        run_btn.pack(pady=20)

        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        messagebox.showerror("Error", f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")

if __name__ == "__main__":
    main_app()
