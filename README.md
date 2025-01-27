# Froshine VoiceCommander: Offline Voice-to-Text IDE Integration

A privacy-focused voice command system for developers that works entirely offline. Uses OpenAI's Whisper AI locally on your machine for speech-to-text and WebRTC VAD for voice activity detection.

Copyright 2025 Adrian Scott

## Features ✨

- 🛡️ **100% Offline** - No audio data leaves your machine
- 🎙️ **Real-time Monitoring** - Continuous voice input detection
- ⌨️ **IDE Integration** - Direct text insertion into code editors
- ✨ **Voice Commands** - Custom commands for common actions
- 🤖 **Command Word Support** - "Flow" prefix for commands (configurable)
- **Pause/Unpause Transcription** - Say "Flow pause" or "Flow unpause"
- 🚀 **GPU Acceleration** - Optional CUDA support for faster processing

## Requirements 📋

- Ubuntu 20.04+ (other Linux distros may work)
- Python 3.8+
- xdotool (`sudo apt install xdotool`)
- PortAudio libraries (`sudo apt install portaudio19-dev`)

## Installation ⚙️

```bash
# Clone repository
git clone https://github.com/AdrianScott/froshine.git
cd froshine

# Install dependencies
sudo apt install portaudio19-dev python3-dev xdotool ffmpeg
pip install -r requirements.txt
```

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

## Configuration

- In voice_monitor_command_word.py, you can modify:

COMMAND_WORD to change your wake word (e.g., "assistant").
COMMANDS dictionary to add or modify commands."

## Troubleshooting 🔧

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

## Privacy & Security 🔒

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
