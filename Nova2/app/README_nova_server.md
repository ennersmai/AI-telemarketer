# Nova Server

A simplified implementation of the Nova speech and conversation system.

## Overview

The Nova Server integrates the following components into a clean, synchronous architecture:

- **Speech-to-Text (STT)**: Transcribes microphone input using Faster Whisper
- **Text-to-Speech (TTS)**: Converts text responses to audio
- **LLM Integration**: Uses language models to generate conversational responses
- **Script Management**: Follows conversation flow based on predefined scripts
- **Context Management**: Tracks conversation history and state

This implementation focuses on simplicity and reliability, avoiding the complexities of async/await patterns when single-threaded processing is sufficient.

## Key Features

- Fully synchronous design for straightforward debugging
- Thread-based processing for non-blocking audio handling
- Event system for component integration
- Clean separation of concerns
- Simple session management

## Running the Server

### Prerequisites

- Python 3.9+
- PyTorch
- Required packages: `faster_whisper`, `sounddevice`, `pydub`

### Setup

1. Ensure you have the Nova2 project in your Python path
2. Configure TTS and LLM credentials as needed (see Configuration section)

### Quick Start

Run the server using the run script:

```bash
python Nova2/app/run_nova.py
```

Or run the server module directly:

```bash
PYTHONPATH="$PYTHONPATH:/path/to/nova_project" python -m Nova2.app.the_server
```

### Command Line Options

- `LOG_LEVEL=DEBUG`: Enable detailed logging
- `MICROPHONE_INDEX=1`: Specify a different microphone (default: 0)
- `FORCE_CPU=1`: Force CPU usage even if CUDA is available

## Configuration

### Speech-to-Text

The default configuration uses the "small" Whisper model. Edit `the_server.py` to change:

```python
stt_conditioning = TranscriptorConditioning(
    model="small",  # Options: tiny, base, small, medium, large
    device="cuda",  # Or "cpu"
    language="en",  # ISO language code or None for auto-detection
    microphone_index=0,
    vad_threshold=0.40,  # Voice activity detection threshold
    voice_boost=3.0  # Speech enhancement level
)
```

### Text-to-Speech

TTS requires configuration with API keys for services like ElevenLabs. Configure this in your environment or before starting the server.

### LLM Integration

The server works with any LLM supported by Nova. Configure the LLM in your environment or before starting the server.

## Architecture

### Components

- **NovaServer**: Main server class that orchestrates all components
- **NovaAPI**: Interface to all Nova components
- **VoiceAnalysis**: Handles STT processing
- **ScriptManager**: Manages conversation flow
- **TTSManager**: Handles text-to-speech conversion
- **AudioPlayer**: Plays audio responses

### Process Flow

1. STT thread continuously processes microphone input
2. When speech is detected, it's transcribed and added to the conversation
3. User input triggers LLM processing in a separate thread
4. LLM generates a response and suggested next state
5. Response is converted to speech and played to the user
6. State is updated based on LLM suggestion
7. Process repeats

## Events

The system uses an event mechanism for component communication:

- `transcription_received`: Fired when new speech transcription is available
- `llm_response_received`: Fired when LLM generates a response
- `tts_completed`: Fired when audio playback finishes
- `state_changed`: Fired when conversation state changes

## Extending

To extend the server:
1. Subclass `NovaServer` to add custom functionality
2. Subscribe to events to add custom behavior
3. Add new components by integrating them in the main class

## Troubleshooting

- **No sound output**: Check audio playback device
- **No speech recognition**: Check microphone index and permissions
- **LLM errors**: Verify API keys and network connectivity
- **Slow performance**: Consider using a smaller model or enabling CUDA

## License

This code is part of the Nova project and is subject to its licensing terms. 