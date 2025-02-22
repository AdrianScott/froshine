import webrtcvad
import pyaudio
import wave
import numpy as np
import torch
import subprocess
import threading
import os
import re
import argparse
from queue import Queue
from signal import signal, SIGINT
import whisper
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from datetime import datetime
import logging
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Default Whisper model (can be overridden)
DEFAULT_WHISPER_MODEL = "small.en"  # Balanced choice for most users
#DEFAULT_WHISPER_MODEL = "large-v3-turbo"  # More accurate but slower

def parse_args():
    # Get model from .env or use default
    default_model = os.getenv('FROSHINE_WHISPER_MODEL', DEFAULT_WHISPER_MODEL)
    
    parser = argparse.ArgumentParser(description='Froshine Voice Commander')
    parser.add_argument('--model', '-m',
                      default=default_model,
                      help='Whisper model to use for speech recognition (default from FROSHINE_WHISPER_MODEL in .env or small.en)')
    return parser.parse_args()

CONFIDENCE_THRESHOLD = 0.42  # Minimum confidence required to use transcription
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
    def __init__(self, model_name):
        logging.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
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

def print_audio_devices():
    """Print detailed information about all audio devices and APIs"""
    audio = pyaudio.PyAudio()
    
    # Print API information
    logging.info("\nAudio APIs available:")
    for i in range(audio.get_host_api_count()):
        api_info = audio.get_host_api_info_by_index(i)
        logging.info(f"API {i}: {api_info['name']}")
        logging.info(f"  Default input device: {api_info.get('defaultInputDevice', 'None')}")
        logging.info(f"  Default output device: {api_info.get('defaultOutputDevice', 'None')}")
    
    # Print all devices
    logging.info("\nAudio devices available:")
    for i in range(audio.get_device_count()):
        try:
            device_info = audio.get_device_info_by_index(i)
            logging.info(f"\nDevice {i}: {device_info['name']}")
            logging.info(f"  API: {device_info['hostApi']}")
            logging.info(f"  Input channels: {device_info['maxInputChannels']}")
            logging.info(f"  Output channels: {device_info['maxOutputChannels']}")
            logging.info(f"  Default sample rate: {device_info['defaultSampleRate']}")
        except Exception as e:
            logging.error(f"Error getting device {i} info: {e}")
    
    return audio

def get_input_device_info():
    """Get audio input device based on system defaults or user configuration.
    
    Environment variables:
    FROSHINE_AUDIO_DEVICE: Name or index of preferred audio device
    FROSHINE_LIST_DEVICES: If set to 1, list all available devices
    """
    audio = pyaudio.PyAudio()
    
    # List devices if requested
    if os.environ.get('FROSHINE_LIST_DEVICES') == '1':
        logging.info("\nAvailable audio devices:")
        for i in range(audio.get_device_count()):
            try:
                device_info = audio.get_device_info_by_index(i)
                logging.info(f"Device {i}: {device_info['name']}")
                logging.info(f"  Input channels: {device_info['maxInputChannels']}")
                logging.info(f"  Sample rate: {device_info['defaultSampleRate']}")
            except Exception:
                continue
    
    # Check for user-specified device
    preferred_device = os.environ.get('FROSHINE_AUDIO_DEVICE')
    if preferred_device is not None:
        try:
            # Try as index first
            if preferred_device.isdigit():
                device_info = audio.get_device_info_by_index(int(preferred_device))
            else:
                # Try as name
                for i in range(audio.get_device_count()):
                    device_info = audio.get_device_info_by_index(i)
                    if preferred_device.lower() in device_info['name'].lower():
                        break
                else:
                    device_info = None
            
            if device_info and device_info.get('maxInputChannels') > 0:
                logging.info(f"Using configured input device: {device_info['name']}")
                return device_info
            else:
                logging.warning(f"Configured device '{preferred_device}' not found or has no input channels")
        except Exception as e:
            logging.warning(f"Error using configured device '{preferred_device}': {e}")
    
    # Try system default
    try:
        device_info = audio.get_default_input_device_info()
        if device_info.get('maxInputChannels') > 0:
            logging.info(f"Using system default input device: {device_info['name']}")
            return device_info
    except Exception as e:
        logging.warning(f"Could not get system default input device: {e}")
    
    # Last resort: find first working input device
    for i in range(audio.get_device_count()):
        try:
            device_info = audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                logging.info(f"Using first available input device: {device_info['name']}")
                return device_info
        except Exception:
            continue
    
    return None

vad = webrtcvad.Vad()
vad.set_mode(3)

audio = pyaudio.PyAudio()
default_device = get_input_device_info()

if not default_device:
    logging.error("No input devices found!")
    raise RuntimeError("No audio input devices available")

try:
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=default_device['index'],
        frames_per_buffer=FRAME_SIZE
    )
    logging.info(f"Successfully opened audio stream with device: {default_device['name']}")
except Exception as e:
    logging.error(f"Failed to open audio stream: {str(e)}")
    raise

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
    if confidence < CONFIDENCE_THRESHOLD:
        logging.info(f"Discarding low-confidence transcription (confidence: {confidence:.2f}): {text}")
        return
        
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

def audio_processor(transcriber):
    global running
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
    args = parse_args()
    signal(SIGINT, signal_handler)
    logging.info(f"Starting Froshine with Whisper model: {args.model}")
    
    print("\nFroshine Incremental Voice Monitor")
    print(f"Wake word: '{COMMAND_WORD}'")
    print("Say:\n  'flow pause' or 'flow unpause' to control transcription.")
    print("  'flow quit' to stop the program;\n  'flow enter' to press Enter\n")
    
    transcriber = Transcriber(args.model)
    recorder_thread = threading.Thread(target=audio_recorder, daemon=True)
    processor_thread = threading.Thread(target=lambda: audio_processor(transcriber), daemon=True)
    
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
    main()