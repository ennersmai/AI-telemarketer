"""
Description: Manages the configuration and running of inference on a TTS model.
"""

import logging
from .inference_engines import InferenceEngineBaseTTS
from .tts_data import TTSConditioning
from .audio_manager import AudioData
from typing import Union, Iterator
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class StreamingAudioData:
    """
    Class for handling streaming audio data chunks.
    Similar to AudioData but provides an iterator interface for streaming.
    """
    def __init__(self, audio_iterator: Iterator[bytes]):
        """
        Initialize the streaming audio data.
        
        Arguments:
            audio_iterator: Iterator that yields audio data chunks.
        """
        self._audio_iterator = audio_iterator
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return next(self._audio_iterator)

class TTSManager:
    def __init__(self) -> None:
        """
        The TTS manager handles all inference related aspects for the TTS inference engines.
        """
        self._inference_engine: InferenceEngineBaseTTS = None
        self._conditioning: TTSConditioning = None
        self._inference_engine_dirty: InferenceEngineBaseTTS = None # Stores config until apply
        self._conditioning_dirty: TTSConditioning = None # Stores config until apply
        print("[DEBUG][TTSManager] __init__ called.")

    def configure(self, inference_engine: InferenceEngineBaseTTS, conditioning: TTSConditioning):
        """
        Provide configuration details for the TTS engine. These are staged until `apply_config` is called.
        """
        print("\n*********** INSIDE TTSManager.configure ***********\n")
        print(f"[DEBUG][TTSManager] configure called. Engine: {type(inference_engine)}, Conditioning: {type(conditioning)}")
        self._inference_engine_dirty = inference_engine
        self._conditioning_dirty = conditioning
        print(f"[DEBUG][TTSManager] configure stored dirty engine: {bool(self._inference_engine_dirty)}")
        print("\n*********** LEAVING TTSManager.configure ***********\n")

    def apply_config(self):
        """
        Applies the staged configuration and loads the model into memory if applicable.
        """
        print("[DEBUG][TTSManager] apply_config called.")
        if self._inference_engine_dirty is None:
            print("[DEBUG][TTSManager] ERROR in apply_config: _inference_engine_dirty is None.")
            raise ValueError("TTS couldn't be configured because no inference engine is configured!")
        
        if self._conditioning_dirty is None:
            print("[DEBUG][TTSManager] ERROR in apply_config: _conditioning_dirty is None.")
            raise ValueError("TTS couldn't be configured because no conditioning data is configured!")
            
        print("[DEBUG][TTSManager] Assigning dirty config to active config...")
        self._inference_engine = self._inference_engine_dirty
        self._conditioning = self._conditioning_dirty
        print(f"[DEBUG][TTSManager] Assigned active engine: {bool(self._inference_engine)}, active conditioning: {bool(self._conditioning)}")

        # Attempt to initialize model in the engine
        try:
            print(f"[DEBUG][TTSManager] Initializing model in engine ({type(self._inference_engine)})...")
            # Pass the active conditioning to initialize_model
            self._inference_engine.initialize_model(self._conditioning) 
            print("[DEBUG][TTSManager] Model initialization successful.")
        except Exception as e:
             print(f"[DEBUG][TTSManager] EXCEPTION during model initialization: {e}")
             logger.error(f"Failed to initialize TTS model in engine: {e}", exc_info=True)
             raise # Re-raise the exception

    def run_inference(self, text: str, stream: bool = False) -> Union[AudioData, StreamingAudioData]:
        """
        Run inference on the TTS.

        Arguments:
            text (str): The text that should be turned into speech.
            stream (bool): Whether to stream the audio data.

        Returns:
            Union[AudioData, StreamingAudioData]: The resulting audio data.
        """
        if self._inference_engine == None:
            raise Exception("TTS is not configured properly.")

        if self._conditioning == None:
            raise Exception("TTS is not configured properly.")

        if not self._inference_engine.is_model_ready():
            self._inference_engine.initialize_model(model=self._conditioning.model)

        # Handle streaming mode
        if stream:
            # Get the audio iterator from the inference engine
            audio_iterator = self._inference_engine.run_inference(text=text, conditioning=self._conditioning, stream=True)
            # Return a StreamingAudioData object that wraps the iterator
            return StreamingAudioData(audio_iterator)
        else:
            # Standard non-streaming mode - needs careful handling
            try:
                # Get the audio from the inference engine
                audio_result = self._inference_engine.run_inference(text=text, conditioning=self._conditioning, stream=False)
                
                # Create an AudioData object
                audio_data = AudioData()
                
                # Handle different return types - some APIs return bytes, others return lists or generators
                if isinstance(audio_result, bytes):
                    # If a single bytes object is returned
                    logging.debug("Received bytes directly from inference engine")
                    audio_data._store_audio([audio_result])
                elif isinstance(audio_result, list) and all(isinstance(item, bytes) for item in audio_result):
                    # If a list of bytes is returned (as expected)
                    logging.debug(f"Received list of {len(audio_result)} bytes items from inference engine")
                    audio_data._store_audio(audio_result)
                elif hasattr(audio_result, '__iter__') and not isinstance(audio_result, (bytes, list)):
                    # If we got an iterator/generator when we didn't expect one
                    logging.debug(f"Received iterator/generator from inference engine in non-streaming mode")
                    try:
                        # Collect all chunks from the generator
                        chunks = []
                        for chunk in audio_result:
                            if isinstance(chunk, bytes):
                                chunks.append(chunk)
                            else:
                                logging.warning(f"Unexpected chunk type: {type(chunk)}")
                                
                        if chunks:
                            logging.debug(f"Collected {len(chunks)} chunks from generator")
                            audio_data._store_audio(chunks)
                        else:
                            raise ValueError("No valid audio chunks found in generator")
                    except Exception as e:
                        logging.error(f"Failed to process audio chunks from generator: {e}")
                        raise Exception(f"Failed to process audio data from generator") from e
                else:
                    # Unexpected data type
                    raise TypeError(f"Unexpected audio data format: {type(audio_result)}")
                
                return audio_data
            except Exception as e:
                logging.error(f"Error in non-streaming TTS: {e}")
                # Create a valid AudioData as a fallback with 1 second of silence
                logging.warning("Creating AudioData with silence as fallback")
                silence = AudioSegment.silent(duration=1000)  # 1 second of silence
                fallback = AudioData()
                fallback._audio_data = silence
                return fallback