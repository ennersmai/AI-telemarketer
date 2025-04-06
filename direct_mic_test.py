import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

# --- Configuration ---
MIC_INDEX = 11
SAMPLE_RATE = 44100
RECORD_DURATION_SECONDS = 10
OUTPUT_DIR = "tmp_output"
OUTPUT_FILENAME = "direct_mic_test_output.wav"
CHANNELS = 1
# --- End Configuration ---

def run_direct_test():
    print("--- Starting Direct Microphone Recording Test ---")
    print(f"Device: {MIC_INDEX}, Rate: {SAMPLE_RATE} Hz, Duration: {RECORD_DURATION_SECONDS}s")

    # Ensure output directory exists
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(script_dir) # Assumes script is in project root
    output_path_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_path_dir, exist_ok=True)
    output_file_path = os.path.join(output_path_dir, OUTPUT_FILENAME)
    print(f"Output will be saved to: {output_file_path}")

    try:
        # Check if device supports the sample rate (optional but good practice)
        sd.check_input_settings(device=MIC_INDEX, channels=CHANNELS, samplerate=SAMPLE_RATE)
        print("Device settings check successful.")

        print(f"Recording for {RECORD_DURATION_SECONDS} seconds...")
        # Record audio directly into a NumPy array
        recording = sd.rec(
            int(RECORD_DURATION_SECONDS * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            device=MIC_INDEX,
            dtype='float32' # Record as float32
        )
        sd.wait() # Wait until recording is finished
        print("Recording finished.")

        # Save the recording to a WAV file
        print(f"Saving recording to {output_file_path}...")
        sf.write(output_file_path, recording, SAMPLE_RATE)
        print("File saved successfully.")

    except sd.PortAudioError as pae:
        print(f"ERROR: PortAudioError during recording: {pae}")
        print(f"Please check if device {MIC_INDEX} supports {CHANNELS} channel(s) at {SAMPLE_RATE} Hz.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

    print("--- Direct Recording Test Finished ---")

if __name__ == "__main__":
    run_direct_test() 