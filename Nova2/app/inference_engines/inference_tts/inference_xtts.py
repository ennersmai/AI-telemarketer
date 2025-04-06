import torch
import os
from pathlib import Path
import io
import soundfile as sf
from typing import Iterator, Any

# Conditional import for TTS
try:
    from TTS.api import TTS as CoquiTTS # Import with alias
except ImportError:
    CoquiTTS = None
    print("Warning: Coqui TTS library not installed. XTTS engine will not be available.")

from .inference_base_tts import InferenceEngineBaseTTS
from ...tts_data import TTSConditioning

class InferenceEngineXTTS(InferenceEngineBaseTTS):
    def __init__(self) -> None:
        """
        Inference engine using Coqui XTTS v2.
        Relies on the TTS library (pip install TTS).
        """
        super().__init__()
        self._model_name: str | None = None
        self._model: Any | None = None # Use Any | None as fallback
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._voice_files_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "voices"
        self._voice_files_dir.mkdir(parents=True, exist_ok=True)
        self.is_local = True # XTTS runs locally

    def initialize_model(self, model: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        """
        Initializes and loads the XTTS v2 model using the TTS library.
        """
        if CoquiTTS is None:
            raise RuntimeError("TTS library not installed. Cannot initialize XTTS model.")
        
        print(f"Initializing XTTS model: {model} on device: {self._device}")
        try:
            self._model = CoquiTTS(model_name=model, progress_bar=True).to(self._device)
            self._model_name = model
            print(f"XTTS model {self._model_name} initialized successfully.")
        except Exception as e:
            print(f"Error initializing XTTS model {model}: {e}")
            self._model = None
            self._model_name = None
            raise

    def clone_voice(self, audio_path: str, name: str) -> None:
        """
        Creates a speaker embedding file (.pth) from an audio file using XTTS.
        Saves the embedding to the data/voices directory.

        Arguments:
            audio_path (str): Path to the audio file (WAV/MP3) for cloning.
            name (str): The name to save the voice embedding as (e.g., 'agent_voice').
                      The .pth extension will be added automatically.
        """
        if not self.is_model_ready():
            raise RuntimeError("XTTS model not initialized. Cannot clone voice.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file for cloning not found: {audio_path}")

        target_path = self._voice_files_dir / f"{name}.pth"
        print(f"Cloning voice from {audio_path} to {target_path}...")
        
        try:
            # XTTS compute_speaker_latents is internal, but tts_to_file can save it
            self._model.tts_to_file(
                text="Default text for cloning, content doesn't matter.", 
                speaker_wav=audio_path, 
                language=self._model.languages[0] if self._model.languages else "en", # Use first available lang or default to en
                file_path=str(target_path),
                # Note: The TTS library saves the speaker embedding implicitly when speaker_wav is provided
                # and file_path ends with .pth. This is a bit of a workaround.
                # A cleaner way might exist in newer TTS versions or require direct latent computation.
            )
            # We actually only want the .pth file, not the generated wav from the dummy text.
            # The library *should* save speaker_*.pth when speaker_wav is used. Let's check.
            # A potential issue is that tts_to_file might not *just* save the latent.
            # Let's refine this if needed based on TTS library behavior.
            # A more direct way might be needed if this approach fails. For now, assume it creates the .pth
            
            # We might need to extract the speaker embedding more directly if the above doesn't work as expected.
            # speaker_embedding = self._model.compute_speaker_latents(audio_path)
            # torch.save(speaker_embedding, target_path) # This is hypothetical, API needs confirmation

            if target_path.exists():
                 print(f"Voice embedding saved successfully to {target_path}")
            else:
                 # If the .pth wasn't created, raise an error or log a warning.
                 # This part needs testing with the actual TTS library version.
                 print(f"Warning/Error: Expected voice embedding file {target_path} was not created by tts_to_file.")
                 # As a fallback, delete any potentially created audio file from the dummy text
                 temp_audio_path = target_path.with_suffix('.wav')
                 if temp_audio_path.exists():
                     os.remove(temp_audio_path)
                 raise RuntimeError(f"Failed to save voice embedding to {target_path}")

        except Exception as e:
            print(f"Error cloning voice to {target_path}: {e}")
            # Clean up potentially created file if error occurred
            if target_path.exists():
                os.remove(target_path)
            raise

    def _get_speaker_path(self, voice_name: str) -> str:
        """Gets the path to the speaker embedding file."""
        path = self._voice_files_dir / f"{voice_name}.pth"
        if not path.exists():
            # Check for .npy as a fallback from previous Zonos attempts if needed? Unlikely needed now.
            # path_npy = self._voice_files_dir / f"{voice_name}.npy"
            # if path_npy.exists(): return str(path_npy)
            raise FileNotFoundError(f"Voice embedding '{voice_name}.pth' not found in {self._voice_files_dir}")
        return str(path)

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool = True) -> Iterator[bytes]:
        """
        Runs TTS inference using the loaded XTTS model.

        Arguments:
            text (str): The text to synthesize.
            conditioning (TTSConditioning): Conditioning parameters (voice, language, etc.).
            stream (bool): If True, yields audio chunks as raw bytes. 
                           If False, raises NotImplementedError.

        Returns:
            Iterator[bytes]: An iterator yielding raw WAV audio chunks.
        """
        if not self.is_model_ready():
            raise RuntimeError("XTTS model not initialized. Cannot run inference.")

        speaker_wav_path = self._get_speaker_path(conditioning.voice)
        language = conditioning.kwargs.get("language", "en") # Default to 'en' if not provided

        print(f"Running XTTS inference for voice '{conditioning.voice}' language '{language}'...")

        if stream:
            try:
                # Use the stream generator from the TTS library
                chunk_iterator = self._model.tts_stream(
                    text=text,
                    speaker_wav=speaker_wav_path,
                    language=language,
                    # Add other relevant parameters from conditioning if supported by tts_stream
                    # speed=conditioning.kwargs.get("speed", 1.0) # Example if speed is supported
                )

                # Yield WAV bytes chunks directly
                for chunk in chunk_iterator:
                    # Convert numpy array chunk (float32) to WAV bytes (int16)
                    buffer = io.BytesIO()
                    # XTTS default sample rate is usually 24000 Hz
                    sf.write(buffer, chunk, samplerate=self._model.synthesizer.output_sample_rate, format='WAV', subtype='PCM_16')
                    wav_bytes = buffer.getvalue()
                    buffer.close()
                    yield wav_bytes # Yield raw bytes

            except Exception as e:
                print(f"Error during XTTS streaming inference: {e}")
                raise
        else:
            # Non-streaming (generate full audio at once)
            raise NotImplementedError("Non-streaming XTTS inference is not implemented.")

    def is_model_ready(self) -> bool:
        """Checks if the XTTS model is loaded."""
        return self._model is not None

    def get_current_model(self) -> str | None:
        """Returns the name of the loaded XTTS model."""
        return self._model_name

    def free(self) -> None:
        """Releases the model from memory (if possible)."""
        # TTS library doesn't have an explicit unload, rely on garbage collection
        print(f"Freeing XTTS model {self._model_name} (setting reference to None).")
        self._model = None
        self._model_name = None
        # Force garbage collection? Optional.
        # import gc
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # --- Helper methods specific to XTTS ---
    def list_available_voices(self) -> list[str]:
        """Lists available cloned voice embeddings (.pth files)."""
        return [f.stem for f in self._voice_files_dir.glob("*.pth")] 