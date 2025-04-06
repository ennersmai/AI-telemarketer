#!/usr/bin/env python
"""
Nova Server Runner

This script provides a convenient way to run the Nova server with proper
environment setup and error handling. Supports command-line configuration.

Usage:
  python run_nova.py [options]

Options:
  --llm <engine>        Specify LLM engine (llamacpp, groq)
  --tts <engine>        Specify TTS engine (zonos, elevenlabs)
  --no-llm              Disable LLM
  --no-tts              Disable TTS
  --no-memory           Disable memory retrieval
  --no-tools            Disable tools
  --mic <index>         Specify microphone index
  --device <device>     Force device (cuda, cpu)
  --debug               Enable debug logging
"""

import os
import sys
import logging
import argparse
from pathlib import Path


def setup_environment(args=None):
    """
    Set up the environment for the Nova server:
    - Add project root to Python path
    - Set environment variables
    - Configure logging

    Args:
        args: Command line arguments
    """
    # Get project root path
    project_root = Path(__file__).parent.parent.parent
    
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_root}"
    
    # Set logging level
    log_level = "DEBUG" if args and args.debug else os.environ.get("LOG_LEVEL", "INFO")
    os.environ["LOG_LEVEL"] = log_level
    
    # Set microphone index
    if args and args.mic is not None:
        os.environ["MICROPHONE_INDEX"] = str(args.mic)
    else:
        os.environ.setdefault("MICROPHONE_INDEX", "0")
    
    # Set device preference
    if args and args.device:
        os.environ["FORCE_CPU"] = "1" if args.device.lower() == "cpu" else "0"
    
    # Set LLM options
    os.environ["USE_LLM"] = "false" if args and args.no_llm else "true"
    if args and args.llm:
        os.environ["LLM_ENGINE"] = args.llm.lower()
    
    # Set TTS options
    os.environ["USE_TTS"] = "false" if args and args.no_tts else "true"
    if args and args.tts:
        os.environ["TTS_ENGINE"] = args.tts.lower()
    
    # Set Memory options
    os.environ["USE_MEMORY"] = "false" if args and args.no_memory else "true"
    
    # Set Tools options
    os.environ["USE_TOOLS"] = "false" if args and args.no_tools else "true"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Environment set up with LOG_LEVEL={log_level}")
    logging.info(f"LLM: {os.environ.get('USE_LLM')} - Engine: {os.environ.get('LLM_ENGINE', 'default')}")
    logging.info(f"TTS: {os.environ.get('USE_TTS')} - Engine: {os.environ.get('TTS_ENGINE', 'default')}")
    logging.info(f"Memory: {os.environ.get('USE_MEMORY')}")
    logging.info(f"Tools: {os.environ.get('USE_TOOLS')}")
    logging.info(f"Using microphone index: {os.environ.get('MICROPHONE_INDEX')}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Nova server")
    
    # LLM options
    parser.add_argument("--llm", choices=["llamacpp", "groq"], 
                        help="Specify LLM engine (llamacpp, groq)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM")
    
    # TTS options
    parser.add_argument("--tts", choices=["zonos", "elevenlabs"], 
                        help="Specify TTS engine (zonos, elevenlabs)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS")
    
    # Memory and tools options
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable memory retrieval")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tools")
    
    # Hardware options
    parser.add_argument("--mic", type=int,
                        help="Specify microphone index")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        help="Force device (cuda, cpu)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def list_available_microphones():
    """List available microphones with their indices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\nAvailable Microphones:")
        print("----------------------")
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(f"Index {idx}: {device['name']}")
        print("\nUse --mic <index> to select a specific microphone\n")
    except Exception as e:
        print(f"Could not list microphones: {e}")


def main():
    """Main function to run the Nova server"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment with arguments
    setup_environment(args)
    
    # List available microphones for user convenience
    list_available_microphones()
    
    # Import the server only after environment is set up
    from Nova2.app.the_server import run_server
    
    try:
        # Run the server
        logging.info("Starting Nova Server...")
        run_server()
        
    except KeyboardInterrupt:
        logging.info("Server shutdown requested by user")
    except Exception as e:
        logging.exception(f"Error running Nova Server: {e}")
    finally:
        logging.info("Nova Server shutdown complete")


if __name__ == "__main__":
    main() 