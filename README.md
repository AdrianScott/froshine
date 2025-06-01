# Froshine VoiceCommander: Offline Voice-to-Text IDE Integration

A privacy-focused voice command system for developers that works entirely offline. Uses OpenAI's Whisper AI locally on your machine for speech-to-text and WebRTC VAD for voice activity detection.

Copyright 2025 Adrian Scott

## Features 

- **100% Offline** - No audio data leaves your machine
- **Real-time Monitoring** - Continuous voice input detection
- **IDE Integration** - Direct text insertion into code editors
- **Voice Commands** - Custom commands for common actions
- **Command Word Support** - "Flow" prefix for commands (configurable)
- **Pause/Unpause Transcription** - Say "Flow pause" or "Flow unpause"
- **GPU Acceleration** - Optional CUDA support for faster processing

## Requirements 

- Ubuntu 20.04+ (other Linux distros may work)
- Python 3.8+
- Working microphone or audio input device
- xdotool (`sudo apt install xdotool` on Ubuntu/Debian)
- PortAudio libraries (`sudo apt install portaudio19-dev`)

## Installation 

1. Clone the repository
2. Install dependencies:
   ```bash
   sudo apt install portaudio19-dev python3-dev xdotool ffmpeg
   pip install -r requirements.txt
   ```

## Configuration 

Froshine can be configured using either:
1. Command-line arguments
2. Environment variables in a `.env` file
3. System environment variables

Command-line arguments take precedence over environment variables.

### Environment File

Copy the example configuration file to create your own:
```bash
cp .env.example .env
```

Then edit `.env` to customize your settings. See `.env.example` for available options.

### Audio Input Configuration

By default, Froshine uses your system's default audio input device. You can configure the audio input using environment variables:

- `FROSHINE_AUDIO_DEVICE`: Specify a preferred audio device by name or index
- `FROSHINE_LIST_DEVICES`: Set to "1" to list all available audio devices

Examples:
```bash
# List all available audio devices
FROSHINE_LIST_DEVICES=1 python voice_monitor_command_word.py

# Use a specific device by name (partial match)
FROSHINE_AUDIO_DEVICE="USB" python voice_monitor_command_word.py

# Use a specific device by index
FROSHINE_AUDIO_DEVICE="2" python voice_monitor_command_word.py
```

### Whisper Model Selection

Froshine supports different Whisper models for speech recognition. You can choose the model using the `--model` or `-m` flag:

```bash
# Use the tiny English model (fastest, less accurate)
python voice_monitor_command_word.py --model tiny.en

# Use the large v3 model (slower, most accurate)
python voice_monitor_command_word.py --model large-v3
```

Available models:
- `tiny.en`: Tiny model (English only) - Fastest, lowest accuracy
- `base.en`: Base model (English only) - Fast, basic accuracy
- `small.en`: Small model (English only) - Default, good balance
- `medium.en`: Medium model (English only) - Better accuracy, slower
- `large-v3`: Large v3 model (All languages) - Best accuracy, slowest

The default model is `small.en`, which provides a good balance between speed and accuracy.

## Usage with voice_monitor_command_word.py

This script continuously listens for voice input, transcribes it locally with Whisper, and types the transcribed text directly into your active window.

Start the script:

```bash
python3 voice_monitor_command_word.py
```

Begin speaking: The system will detect speech and automatically type the transcribed text into your currently focused application.

**Issue commands:**

- Say "Flow enter" to press Enter.
- Say "Flow save file" to simulate Ctrl+S.
- Say "Flow pause" to stop typing text (commands still work).
- Say "Flow unpause" to resume typing text.
- Stop the script: Say "Flow quit", or press Ctrl+C in the terminal to exit.

## Troubleshooting 

**Common Issues:**

- **ALSA/JACK warnings**: Normal and safe to ignore
- **No audio input**:
  ```bash
  # Check recording devices
  arecord -l
  ```
- **Permission issues**:
  ```bash
  sudo usermod -a -G audio $USER
  # Reboot after running
  ```

## Privacy & Security 

- All audio processing happens locally
- No internet connection required
- No tracking or data collection

## Copyright

Copyright 2025 Adrian Scott

---

**Acknowledgements**:

- OpenAI Whisper AI model
- WebRTC VAD for voice detection
- PyAudio for audio capture

````
early work in progress

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

sudo apt install xdotool
````

```
WINDOW_ID=$(xdotool search --name "\(Workspace\) \- Windsurf")
echo $WINDOW_ID
xdotool windowactivate --sync $WINDOW_ID; xdotool type --window $WINDOW_ID --delay 0 "windsurf test froshine"

```

Current mechanism is to start voice recorder, voice_to_ide.sh, then click in the field of Windsurf I want it to go into.

Next step: voice detection to automatically fire up the recorder.

After that: voice commands to choose window, and especially use Freepoprompt and o1-xml-parser to update files.
