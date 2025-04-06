import unittest
import time
import os
import logging
import sys
import asyncio
from pathlib import Path

# Adjust import path assuming the test is run from the project root
# Add project root to sys.path to find Nova2 package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from Nova2.app.transcriptor import VoiceAnalysis
    from Nova2.app.transcriptor_data import TranscriptorConditioning
    from Nova2.app.context_data import ContextDatapoint, ContextSource_Voice
except ImportError as e:
    print(f"Error importing Nova2 components: {e}")
    print("Ensure you are running the test from the project root directory (`nova_project`)")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (match server local test where possible) ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
MIC_INDEX = 10 # Hardcoded index used in server's local test
VAD_THRESHOLD = 0.20
VOICE_BOOST = 3.0
TEST_DURATION_SECONDS = 20 # How long to run the test listening for STT output
# --- End Configuration ---

class TestVoiceAnalysisSTT(unittest.TestCase):

    def setUp(self):
        """Set up the VoiceAnalysis instance before each test."""
        logger.info("Setting up VoiceAnalysis for test...")
        self.voice_analysis = VoiceAnalysis()

        # --- Define Conditioning ---
        device_to_use = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
        logger.info(f"Using device: {device_to_use}")

        self.conditioning = TranscriptorConditioning(
            model=WHISPER_MODEL_SIZE,
            device=device_to_use,
            language="en",
            microphone_index=MIC_INDEX,
            vad_threshold=VAD_THRESHOLD,
            voice_boost=VOICE_BOOST
        )
        logger.info(f"Conditioning: Model={self.conditioning.model}, Mic={self.conditioning.microphone_index}, VAD={self.conditioning.vad_threshold}, Boost={self.conditioning.voice_boost}")

        # --- Configure and Apply ---
        try:
            self.voice_analysis.configure(self.conditioning)
            self.voice_analysis.apply_config()
            logger.info("VoiceAnalysis configured and config applied.")
        except Exception as e:
            logger.error(f"Error during setUp configuration: {e}", exc_info=True)
            self.fail(f"Setup failed: {e}") # Fail the test if setup has issues


    def tearDown(self):
        """Clean up resources after each test."""
        logger.info("Tearing down VoiceAnalysis...")
        try:
            if hasattr(self, 'voice_analysis') and self.voice_analysis:
                self.voice_analysis.close()
                logger.info("VoiceAnalysis closed.")
        except Exception as e:
            logger.error(f"Error during tearDown: {e}", exc_info=True)

    def test_stt_yields_datapoints(self):
        """Test if the start() generator yields at least one datapoint within the time limit."""
        logger.info(f"Starting test: Listening for STT output for {TEST_DURATION_SECONDS} seconds...")
        datapoint_count = 0
        start_time = time.time()
        stt_output_received = False

        try:
            # The start method in the current transcriptor.py doesn't accept input sample rate
            stt_generator = self.voice_analysis.start()

            while time.time() - start_time < TEST_DURATION_SECONDS:
                try:
                    # Use a timeout mechanism for getting the next item
                    # Note: This assumes the generator doesn't block indefinitely if queue is empty
                    # A more robust approach might involve running the generator
                    # in a separate thread/task and checking a shared queue.
                    datapoint = next(stt_generator)
                    logger.info(f"Received datapoint: {type(datapoint)}")
                    if isinstance(datapoint, ContextDatapoint):
                         logger.info(f"  -> Content: '{datapoint.content}'")
                         datapoint_count += 1
                         stt_output_received = True
                         # Optional: Break early if we receive output
                         # break
                    # Add a small sleep to prevent hogging CPU if generator yields very fast
                    time.sleep(0.05)

                except StopIteration:
                    logger.info("STT generator finished (StopIteration).")
                    break
                except Exception as gen_err:
                     logger.error(f"Error fetching from STT generator: {gen_err}", exc_info=True)
                     # Decide whether to break or continue on generator error
                     break

            elapsed_time = time.time() - start_time
            logger.info(f"Test finished after {elapsed_time:.2f} seconds. Total datapoints received: {datapoint_count}")

        except Exception as e:
            logger.error(f"Exception during test execution: {e}", exc_info=True)
            self.fail(f"Test execution failed: {e}")

        self.assertGreater(datapoint_count, 0, f"STT did not yield any datapoints within {TEST_DURATION_SECONDS} seconds.")
        # Or use assertTrue if breaking early:
        # self.assertTrue(stt_output_received, f"STT did not yield any datapoints within {TEST_DURATION_SECONDS} seconds.")

if __name__ == '__main__':
    logger.info("Running STT Unit Test...")
    unittest.main() 