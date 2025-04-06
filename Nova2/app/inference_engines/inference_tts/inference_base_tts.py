"""
Description: Provides a base class for all TTS inference engines to ensure a consistent structure and documentation.
"""
from typing import Union, List, Iterator
from ...tts_data import TTSConditioning

class InferenceEngineBaseTTS:
    """
    Provides a base class for all TTS inference engines to ensure a consistent structure.
    """
    def __init__(self):
        self._type = "TTS"
        self.is_local = False

    def initialize_model(self, conditioning: TTSConditioning) -> None:
        """
        Load the model into VRAM/RAM based on the provided conditioning.
        Required to run inference. Call free() to free up the VRAM/RAM again.
        
        Arguments:
            conditioning (TTSConditioning): The full conditioning object containing model ID and other relevant settings.
        """
        pass

    def select_model_async(self):
        """
        Load the model into VRAM/RAM. Required to run inference. Call free() to free up the VRAM/RAM again. Runs async.
        """
        pass

    def select_voice(self):
        """
        Set the voice to be used.
        """
        pass

    def is_model_ready(self) -> bool:
        """
        Checks if the engine is ready to run inference.
        
        Returns:
            bool: True if the model is ready, False otherwise
        """
        pass

    def get_current_model(self) -> str:
        """
        Which model is currently loaded.
        
        Returns:
            str: The name of the currently loaded model
        """
        pass

    def free(self) -> None:
        """
        Frees the VRAM/RAM. The model can not be used anymore after it was freed. 
        It needs to be loaded again by calling initialize_model().
        """
        pass

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool = False) -> Union[List[bytes], Iterator[bytes]]:
        """
        Get the spoken text from the TTS.
        
        Arguments:
            text (str): The text to convert to speech
            conditioning (TTSConditioning): Parameters for conditioning the TTS generation
            stream (bool): Whether to stream the audio data (default: False)
            
        Returns:
            Union[List[bytes], Iterator[bytes]]: Either a list of audio data (non-streaming) 
                                               or an iterator of audio chunks (streaming)
        """
        pass