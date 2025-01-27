import webrtcvad
import pyaudio
import wave
import numpy as np
import torch
import subprocess
import threading
import os
import re
from queue import Queue
from signal import signal, SIGINT
import whisper
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from datetime import datetime
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_commands.log'),
        logging.StreamHandler()
    ]
)

# Configuration for faster, more frequent chunking
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 20   # 20ms frames
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
POST_SPEECH_BUFFER = 400
FRAMES_TO_KEEP_AFTER_SILENCE = int(POST_SPEECH_BUFFER / FRAME_DURATION)

COMMAND_WORD = "flow"

COMMAND_SYNONYMS = {
    "pause": ["pause", "paws", "paus", "pawz"],
    "unpause": ["unpause", "onpause", "on pause", "un pause"],
    "enter": ["enter", "inner"],
    "quit": ["quit", "quick"],
    "switch to browser": ["switch to browser", "open browser"],
    "save file": ["save file", "save document"]
}

COMMANDS = {
    "enter": ["enter"],
    "switch to browser": ["switch to browser"],
    "save file": ["save file"],
    "pause": ["pause"],
    "unpause": ["unpause"],
    "quit": ["quit"],
}

running = True
typed_history = ""
is_paused = False

# We'll store a pending command word if the last chunk ended with "flow"
pending_command_word = None

class Transcriber:
    def __init__(self):
        self.model = whisper.load_model("medium.en")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def transcribe(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            fp16=(self.device == "cuda"),
            language="en",
            verbose=False
        )
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(seg.get("avg_logprob", 0) for seg in segments) / len(segments)
            confidence = math.exp(avg_logprob)
        else:
            confidence = 0.0
        return result["text"].strip(), confidence

def get_audio_data(audio_path):
    try:
        with wave.open(audio_path, 'rb') as wf:
            return wf.readframes(wf.getnframes())
    except Exception as e:
        logging.error(f"Failed to load {audio_path}: {str(e)}")
        return None

def log_transcription(transcription, is_command=False, confidence=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': 'command' if is_command else 'transcription',
        'content': transcription,
    }
    if confidence is not None:
        log_entry['confidence'] = f"{confidence:.2f}"
    logging.info(log_entry)

def clean_audio(input_path, output_path):
    try:
        audio = AudioSegment.from_wav(input_path)
        audio = low_pass_filter(audio, 4000)
        audio = normalize(audio)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"Audio cleaning error: {e}")
        return False

vad = webrtcvad.Vad()
vad.set_mode(3)

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_SIZE
)

audio_queue = Queue()

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False
    audio_queue.put(None)

signal(SIGINT, signal_handler)

def save_audio(frames, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def execute_command(command):
    global running, is_paused
    if command == "quit":
        print("\nQuitting voice commander...")
        logging.info("User initiated graceful shutdown")
        running = False
        return
    elif command == "enter":
        subprocess.run(["xdotool", "key", "Return"])
    elif command == "switch to browser":
        subprocess.run(["xdotool", "search", "--name", "Browser", "windowactivate"])
    elif command == "save file":
        subprocess.run(["xdotool", "key", "ctrl+s"])
    elif command == "pause":
        is_paused = True
        print("Transcription paused.")
    elif command == "unpause":
        is_paused = False
        print("Transcription resumed.")
    else:
        print(f"Unknown command: {command}")

def normalize_word(word: str) -> str:
    # e.g. "Flow," -> "flow", "Flowpaws" -> "flowpaws"
    return re.sub(r'[^a-z0-9]+', '', word.lower())

def find_command_for_synonym(syn: str) -> str:
    for cmd, synonyms_list in COMMAND_SYNONYMS.items():
        if syn in synonyms_list:
            if cmd in COMMANDS:
                return cmd
    return ""

def process_audio_chunk(frames, transcriber):
    if not frames:
        return
    raw_path = "temp_raw.wav"
    clean_path = "temp_clean.wav"
    save_audio(frames, raw_path)
    clean_success = clean_audio(raw_path, clean_path)
    audio_path = clean_path if clean_success else raw_path
    audio_data = get_audio_data(audio_path)
    if not audio_data:
        return

    text, confidence = transcriber.transcribe(audio_path)
    process_transcription_text(text, confidence)

def process_transcription_text(text: str, confidence: float):
    """
    Splits text, ensures a command only executes if preceded by "flow" 
    in the same chunk or via pending_command_word from the last chunk.
    """
    global typed_history, is_paused, pending_command_word

    words = text.split()
    # Check if we have a pending command word from last chunk
    in_command_scope = False
    if pending_command_word:
        # If there's at least one word, try combining
        if words:
            first_norm = normalize_word(words[0])
            combined = pending_command_word + first_norm
            is_cmd, recognized = interpret_potential_command(combined)
            if is_cmd:
                # remove first from words
                log_transcription(f"{pending_command_word} {words[0]}", is_command=True, confidence=confidence)
                execute_command(recognized)
                words.pop(0)
                in_command_scope = False
            else:
                # If not recognized, type the pending word if not paused
                if not is_paused:
                    type_text(pending_command_word)
        else:
            # No new words, just type the pending word
            if not is_paused:
                type_text(pending_command_word)
        pending_command_word = None

    typed_words = []
    # We'll track if we see the command word
    for i, w in enumerate(words):
        norm = normalize_word(w)
        # If we see the exact wake word and it's not the last word, we are in command scope for subsequent words
        if norm == COMMAND_WORD and i < len(words) - 1:
            in_command_scope = True
            continue

        # If we see the exact wake word and it's the last word in chunk, store pending
        if norm == COMMAND_WORD and i == len(words) - 1:
            pending_command_word = COMMAND_WORD
            continue

        # Otherwise, see if this might be a command
        if in_command_scope:
            is_cmd, recognized = interpret_potential_command(norm)
            if is_cmd:
                log_transcription(w, is_command=True, confidence=confidence)
                execute_command(recognized)
                # once we do a command, we leave command scope
                in_command_scope = False
            else:
                # If it's not a recognized command, we type it
                typed_words.append(w)
                # leave command scope after one attempt, so "flow hello quit" won't trigger quit
                in_command_scope = False
        else:
            typed_words.append(w)

    # Type typed_words if not paused
    if not is_paused and typed_words:
        joined = " ".join(typed_words)
        log_transcription(joined, confidence=confidence)
        type_text(joined)

def interpret_potential_command(norm_word: str) -> (bool, str):
    """
    Returns (is_command, command_key).
    e.g. "pause" or "paws" => (True, "pause") if in command scope
    """
    # If word starts with "flow" + remainder, we ignore that unless user specifically said "flow <command>"
    # So let's say: if the user is in command scope, we only check synonyms
    cmd_key = find_command_for_synonym(norm_word)
    if cmd_key:
        return True, cmd_key
    return False, ""

def type_text(text: str):
    global typed_history
    prefix = ""
    if typed_history and typed_history[-1] in {'.', '!', '?'}:
        prefix = " "
    subprocess.run(["xdotool", "type", "--delay", "0", prefix + text])
    typed_history += prefix + text

def audio_recorder():
    print("Audio recorder started...")
    while running:
        try:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            audio_queue.put(frame)
        except:
            break

def audio_processor():
    global running
    transcriber = Transcriber()
    print("Audio processor started...")

    speech_frames = []
    silent_frames_count = 0
    is_speaking = False

    while running:
        frame = audio_queue.get()
        if frame is None:
            break
        if vad.is_speech(frame, RATE):
            if not is_speaking:
                speech_frames = []
                silent_frames_count = 0
                is_speaking = True
            speech_frames.append(frame)
        else:
            if is_speaking:
                silent_frames_count += 1
                if silent_frames_count > FRAMES_TO_KEEP_AFTER_SILENCE:
                    process_audio_chunk(speech_frames, transcriber)
                    speech_frames = []
                    silent_frames_count = 0
                    is_speaking = False

def main():
    print("\nFroshine Incremental Voice Monitor")
    print(f"Wake word: '{COMMAND_WORD}'")
    print("Say 'flow pause' or 'flow unpause' to control transcription.\n")

    recorder_thread = threading.Thread(target=audio_recorder, daemon=True)
    processor_thread = threading.Thread(target=audio_processor, daemon=True)
    recorder_thread.start()
    processor_thread.start()

    try:
        while running:
            recorder_thread.join(0.1)
            processor_thread.join(0.1)
    finally:
        print("Cleaning up resources...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if os.path.exists("temp_raw.wav"):
            os.remove("temp_raw.wav")
        if os.path.exists("temp_clean.wav"):
            os.remove("temp_clean.wav")
        print("Exit complete")

if __name__ == "__main__":
    signal(SIGINT, signal_handler)
    main()