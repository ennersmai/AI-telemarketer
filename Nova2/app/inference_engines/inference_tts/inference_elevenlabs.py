from typing import Literal, Union, List, Iterator

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from elevenlabs import stream as play_stream

from .inference_base_tts import InferenceEngineBaseTTS
from ... import security_manager
from ...security_data import Secrets
from ...tts_data import TTSConditioning

class InferenceEngineElevenlabs(InferenceEngineBaseTTS):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via the elevenlabs API.
        """
        self._key_manager = security_manager.SecretsManager()

        self._model = None

        super().__init__()

        self.is_local = False

    def initialize_model(self, conditioning: TTSConditioning) -> None:
        """
        Initializes the ElevenLabs client and stores the model ID from conditioning.
        
        Arguments:
            conditioning (TTSConditioning): The conditioning object containing model and other settings.
        """
        print("[DEBUG][InferenceEngineElevenlabs] initialize_model called.")
        key = self._key_manager.get_secret(Secrets.ELEVENLABS_API)

        if not key:
            print("[DEBUG][InferenceEngineElevenlabs] ERROR: API key not found.")
            raise ValueError("Elevenlabs API key not found")
        
        print("[DEBUG][InferenceEngineElevenlabs] API key found. Initializing client...")
        self._elevenlabs_client = ElevenLabs(
            api_key=key
        )
        
        self._model = conditioning.model 
        print(f"[DEBUG][InferenceEngineElevenlabs] Stored model ID: {self._model}")

    def is_model_ready(self) -> bool:
        ready = bool(self._elevenlabs_client and self._model)
        print(f"[DEBUG][InferenceEngineElevenlabs] is_model_ready called. Returning: {ready}")
        return ready
    
    def get_current_model(self) -> str:
        return self._model
    
    def free(self) -> None:
        self._model = None
        self._elevenlabs_client = None
        print("[DEBUG][InferenceEngineElevenlabs] free called. Model and client cleared.")

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool) -> Union[List[bytes], Iterator[bytes]]:
        """
        Run inference on the ElevenLabs TTS engine with fallback logic.
        
        Arguments:
            text (str): The text to generate speech for
            conditioning (TTSConditioning): The conditioning parameters (Note: Model ID from init is used)
            stream (bool): Whether to stream the audio data
            
        Returns:
            Union[List[bytes], Iterator[bytes]]: Either a list of audio data (non-streaming) or
                                                an iterator of audio chunks (streaming)
                                                
        Raises:
            RuntimeError: If both primary and fallback API calls fail.
        """
        if not self.is_model_ready():
             print("[DEBUG][InferenceEngineElevenlabs] ERROR: run_inference called but model not ready.") 
             raise RuntimeError("ElevenLabs model is not initialized. Call initialize_model first.")

        print(f"[DEBUG][InferenceEngineElevenlabs] run_inference called. Model: {self._model}, Voice: {conditioning.voice}, Requested Stream: {stream}")

        voice = Voice(
            voice_id=conditioning.voice,
            settings=VoiceSettings(
                stability=conditioning.stability,
                similarity_boost=conditioning.similarity_boost,
                style=conditioning.expressivness,
                use_speaker_boost=conditioning.use_speaker_boost
            )
        )
        
        primary_error = None
        fallback_error = None

        # --- Try 1: Preferred API (Modern convert/convert_as_stream) ---
        try:
            print(f"[DEBUG][InferenceEngineElevenlabs] Attempting modern API (Stream={stream})...")
            if stream:
                print("[DEBUG][InferenceEngineElevenlabs] > Calling convert_as_stream...") 
                result = self._elevenlabs_client.text_to_speech.convert_as_stream(
                    text=text,
                    voice_id=conditioning.voice,
                    model_id=self._model, 
                    voice_settings=voice.settings
                )
                print("[DEBUG][InferenceEngineElevenlabs] > convert_as_stream succeeded.")
                return result # Success!
            else:
                print("[DEBUG][InferenceEngineElevenlabs] > Calling convert...") 
                audio_data_result = self._elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=conditioning.voice,
                    model_id=self._model, 
                    voice_settings=voice.settings
                )
                # --- Handle potential generator return from convert --- 
                audio_bytes_list = []
                if hasattr(audio_data_result, '__iter__') and not isinstance(audio_data_result, bytes):
                    print("[DEBUG][InferenceEngineElevenlabs] > Convert returned iterator, collecting chunks...")
                    for chunk in audio_data_result:
                        if isinstance(chunk, bytes):
                            audio_bytes_list.append(chunk)
                    if not audio_bytes_list:
                         raise ValueError("Modern API (convert) iterator yielded no audio data.")
                elif isinstance(audio_data_result, bytes):
                     print("[DEBUG][InferenceEngineElevenlabs] > Convert returned bytes directly.")
                     audio_bytes_list = [audio_data_result]
                else:
                     raise TypeError(f"Unexpected return type from convert: {type(audio_data_result)}")
                # --- End handle generator --- 
                     
                print(f"[DEBUG][InferenceEngineElevenlabs] > Convert finished. Collected audio bytes length: {sum(len(b) for b in audio_bytes_list)}") 
                return audio_bytes_list # Success! (Return list of bytes)
        except Exception as e:
            primary_error = e
            print(f"[DEBUG][InferenceEngineElevenlabs] Modern API failed (Stream={stream}): {e}")

        # --- Try 2: Fallback API (Older generate) --- 
        if primary_error:
            try:
                print(f"[DEBUG][InferenceEngineElevenlabs] Attempting fallback generate API (Stream={stream})...")
                result = self._elevenlabs_client.generate(
                    text=text,
                    voice=voice,
                    model=self._model, 
                    stream=stream
                )
                print("[DEBUG][InferenceEngineElevenlabs] Fallback generate API finished.") 
                
                # Check if non-streaming fallback actually returned data
                if not stream:
                    audio_list = list(result)
                    if not audio_list or not any(audio_list):
                        raise ValueError("Fallback API (generate) returned empty audio data.")
                    print(f"[DEBUG][InferenceEngineElevenlabs] > Fallback non-stream successful. Data length (approx): {sum(len(c) for c in audio_list)}")
                    return audio_list # Success!
                else:
                    print("[DEBUG][InferenceEngineElevenlabs] > Fallback stream successful.")
                    return result # Success!
                    
            except Exception as e:
                fallback_error = e
                print(f"[DEBUG][InferenceEngineElevenlabs] Fallback generate API failed (Stream={stream}): {e}")

        # --- Both Attempts Failed --- 
        error_message = f"ElevenLabs TTS failed after multiple attempts. Primary error: {primary_error}. Fallback error: {fallback_error}"
        print(f"[ERROR][InferenceEngineElevenlabs] {error_message}")
        raise RuntimeError(error_message)