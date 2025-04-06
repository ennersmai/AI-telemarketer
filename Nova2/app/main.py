#!/usr/bin/env python
"""
Nova Server Main Entry Point

This script starts the Nova Server with proper environment configuration.
Run with `python -m Nova2.app.main` from the project root.

Command-line options are available to customize the server behavior.
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
    - Set environment variables based on arguments
    - Configure logging
    
    Args:
        args: Optional command line arguments
    """
    # Get project root path
    project_root = Path(__file__).parent.parent.parent
    
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_root}"
    
    # Set log level
    log_level = "DEBUG" if args and args.debug else os.environ.get("LOG_LEVEL", "INFO")
    os.environ["LOG_LEVEL"] = log_level
    
    # Set default engines
    os.environ.setdefault("LLM_ENGINE", "groq")
    os.environ.setdefault("TTS_ENGINE", "elevenlabs")
    
    # Set component flags
    if args:
        if args.no_llm:
            os.environ["USE_LLM"] = "false"
        if args.no_tts:
            os.environ["USE_TTS"] = "false"
        if args.no_memory:
            os.environ["USE_MEMORY"] = "false"
        if args.no_tools:
            os.environ["USE_TOOLS"] = "false"
        
        # Set specific engines if requested
        if args.llm_engine:
            os.environ["LLM_ENGINE"] = args.llm_engine
        if args.tts_engine:
            os.environ["TTS_ENGINE"] = args.tts_engine
        
        # Set microphone index
        if args.mic_index is not None:
            os.environ["MICROPHONE_INDEX"] = str(args.mic_index)
        else:
            os.environ.setdefault("MICROPHONE_INDEX", "11")
            
        # Set device preference
        if args.force_cpu:
            os.environ["FORCE_CPU"] = "1"
            
        # Set script mode
        if args.telemarketer:
            os.environ["SCRIPT_MODE"] = "telemarketer"
    else:
        # Default settings
        os.environ.setdefault("MICROPHONE_INDEX", "11")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Environment set up with LOG_LEVEL={log_level}")
    
    # Log configuration
    logging.info(f"LLM: {os.environ.get('USE_LLM', 'true')} - Engine: {os.environ.get('LLM_ENGINE', 'default')}")
    logging.info(f"TTS: {os.environ.get('USE_TTS', 'true')} - Engine: {os.environ.get('TTS_ENGINE', 'default')}")
    logging.info(f"Memory: {os.environ.get('USE_MEMORY', 'true')}")
    logging.info(f"Tools: {os.environ.get('USE_TOOLS', 'true')}")
    logging.info(f"Microphone index: {os.environ.get('MICROPHONE_INDEX')}")
    if args and args.telemarketer:
        logging.info("Running in telemarketer script mode")


def parse_arguments():
    """Parse command line arguments for Nova server configuration."""
    parser = argparse.ArgumentParser(description="Start the Nova server")
    
    # Component options
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM functionality")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS functionality")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory functionality")
    parser.add_argument("--no-tools", action="store_true", help="Disable tools functionality")
    
    # Engine selection
    parser.add_argument("--llm-engine", choices=["llamacpp", "groq"], 
                       help="Specify LLM engine to use")
    parser.add_argument("--tts-engine", choices=["zonos", "elevenlabs"], 
                       help="Specify TTS engine to use")
    
    # Hardware options
    parser.add_argument("--mic-index", type=int, help="Specify microphone index to use")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    
    # Script mode options
    parser.add_argument("--telemarketer", action="store_true", 
                       help="Run with telemarketer script for making outbound sales calls")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()


def main():
    """Main entry point to run the Nova server"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment with arguments
    setup_environment(args)
    
    # Import the appropriate server implementation based on script mode
    if os.environ.get("SCRIPT_MODE") == "telemarketer":
        from Nova2.app.the_server import run_server
        logging.info("Starting Nova Server in telemarketer mode...")
    else:
        from Nova2.app.the_server import run_server
        logging.info("Starting Nova Server in standard mode...")
    
    try:
        # Start the server
        run_server()
        
    except KeyboardInterrupt:
        logging.info("Server shutdown requested by user")
    except Exception as e:
        logging.exception(f"Error running Nova Server: {e}")
    finally:
        logging.info("Nova Server shutdown complete")


if __name__ == "__main__":
    main() 