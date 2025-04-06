"""
Description: This script uses fasterwhisper to continously transcribe audio data from the microphone. It also creates voice embeddings.
"""

import os
import queue
import threading
import time
from typing import Generator, List
import multiprocessing
import logging

from . import helpers

import langcodes
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from denoiser import pretrained
from denoiser.dsp import convert_audio
from faster_whisper import WhisperModel
with helpers.suppress_output():
    from speechbrain.inference.speaker import EncoderClassifier
import silero_vad

from .transcriptor_data import Word, TranscriptorConditioning
from .database_manager import VoiceDatabaseManager
from .context_data import *

SAMPLE_RATE = 16000

# Define logger for this module
logger = logging.getLogger(__name__)

class VoiceAnalysis:
    def __init__(self) -> None:
        """
        A pipeline for live voice analysis.
        """
        self._conditioning = None
        self._conditioning_dirty = None

        if os.name == "nt":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Only necessary for windows

        self._voice_database_manager = VoiceDatabaseManager()

    def configure(self, conditioning: TranscriptorConditioning):
        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        if self._conditioning_dirty is None:
            raise Exception("Failed to initialize TTS. No TTS conditioning provided.")

        self._conditioning = self._conditioning_dirty
        self._verbose = False

        # Set the device
        self._device = self._conditioning.device
        if self._conditioning.device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        logger.info(f"Using device '{self._device}' for speech processing.")
        torch.set_default_dtype(torch.float32)

        # Initialize whisper model
        try:
            logger.info(f"Loading Whisper model '{self._conditioning.model}'...")
            with helpers.suppress_output():
                self._model = WhisperModel(
                    model_size_or_path=self._conditioning.model,
                    device=self._device,
                    compute_type="float32",
                    cpu_threads=multiprocessing.cpu_count()
                )
            logger.info(f"Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
            
        # Set up language
        if self._conditioning.language is not None:
            try:
                langcodes.Language.get(self._conditioning.language)
                logger.info(f"Language set to '{self._conditioning.language}'.")
            except Exception:
                logger.error(f"{self._conditioning.language} is not a valid language code.")
                raise ValueError(f"{self._conditioning.language} is not a valid language code.")
        else:
            logger.info("Language set to auto (will be detected by Whisper).")
        self._language = self._conditioning.language
        
        # Initialize denoiser if needed
        if self._conditioning.voice_boost != 0.0:
            try:
                logger.info("Loading denoiser model...")
                self._denoise_model = pretrained.dns64()
                self._denoise_model.eval()
                self._denoise_model = self._denoise_model.to(self._device)
                logger.info("Denoiser model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load denoiser model: {str(e)}")
                self._conditioning.voice_boost = 0.0
        else:
            logger.info("Voice boost is set to 0, skipping denoiser model loading.")
        
        # Initialize VAD model
        try:
            logger.info("Loading VAD model...")
            vad_result = silero_vad.load_silero_vad()
            # Handle the result without using tuple unpacking which may cause __iter__ NotImplementedError
            if isinstance(vad_result, tuple) and len(vad_result) >= 1:
                self._vad_model = vad_result[0]
                logger.info("VAD model extracted from result tuple")
            else:
                self._vad_model = vad_result
                logger.info("VAD model loaded as single object")
            
            # Move the model to the desired device
            if hasattr(self._vad_model, 'to') and callable(getattr(self._vad_model, 'to')):
                self._vad_model = self._vad_model.to(self._device)
                logger.info(f"VAD model moved to device: {self._device}")
            logger.info("VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {str(e)}")
            raise
        
        # Initialize speaker embedding model
        try:
            logger.info("Loading speaker embedding model...")
            with helpers.suppress_output():
                self._speaker_embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb", 
                    savedir="pretrained_models/spkrec-xvect-voxceleb", 
                    run_opts={"device": self._device}
                )
            logger.info("Speaker embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load speaker embedding model: {str(e)}")
            raise

        # Set up recording parameters
        self._microphone_index = self._conditioning.microphone_index
        logger.info(f"Using microphone index: {self._microphone_index}")
        self._max_silence_chunks = 3
        self._current_sentence = []
        self._locked_words = 0
        self._audio_queue = queue.Queue()
        self._is_recording = True
        self._voice_boost = self._conditioning.voice_boost
        
        # Create but don't start recording thread yet
        self._recording_thread = threading.Thread(target=self._record_audio, name="AudioRecorder")
        self._recording_thread.daemon = True  # Make thread a daemon so it doesn't prevent program exit
        
        # Give a small delay to ensure all models are fully loaded
        time.sleep(0.5)
        
        logger.info("VoiceAnalysis initialization complete.")

    def _record_audio(self) -> None:
        """Records audio from the microphone and puts it into the queue."""
        audio_buffer = np.array([], dtype=np.float32)  
        last_transcription_time = time.time()
        thread_id = threading.get_ident()
        logger.info(f"[{thread_id}] Recording thread started with microphone index {self._microphone_index}")

        def callback(indata, frames, time_info, status):
            nonlocal audio_buffer, last_transcription_time
            if status:
                logger.warning(f"[{thread_id}] Audio callback status: {status}")
                
            # Convert input data to float32 array
            audio = np.frombuffer(indata, dtype=np.float32)  
            
            # Concatenate with buffer
            audio_buffer = np.concatenate((audio_buffer, audio))

            # Put accumulated audio in queue about once per second
            if time.time() - last_transcription_time >= 1:
                last_transcription_time = time.time()
                if len(audio_buffer) > 0:
                    try:
                        self._audio_queue.put(audio_buffer)
                        logger.debug(f"[{thread_id}] Added {len(audio_buffer)} samples to audio queue")
                        audio_buffer = np.array([], dtype=np.float32)
                    except Exception as e:
                        logger.error(f"[{thread_id}] Error putting audio in queue: {str(e)}")

        try:
            logger.info(f"[{thread_id}] Opening input stream with device {self._microphone_index}")
            with sd.InputStream(
                callback=callback, 
                dtype=np.float32, 
                channels=1, 
                samplerate=SAMPLE_RATE, 
                device=self._microphone_index,
                blocksize=4096  # Use a reasonable blocksize
            ) as stream:
                logger.info(f"[{thread_id}] Audio stream opened successfully")
                
                # Main recording loop
                while self._is_recording:
                    time.sleep(0.1)
                
                logger.info(f"[{thread_id}] Recording stopped (_is_recording became False)")
                
        except Exception as e:
            logger.error(f"[{thread_id}] Error in recording thread: {str(e)}", exc_info=True)
        finally:
            # Put any remaining audio in the queue
            if len(audio_buffer) > 0:
                try:
                    self._audio_queue.put(audio_buffer)
                    logger.debug(f"[{thread_id}] Added final {len(audio_buffer)} samples to audio queue")
                except Exception as e:
                    logger.error(f"[{thread_id}] Error putting final audio in queue: {str(e)}")
                    
            logger.info(f"[{thread_id}] Recording thread exiting")

    def _boost_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """Boost the speech signal in the audio."""
        if self._voice_boost == 0.0:
            return audio_data

        try:
            audio_tensor = torch.from_numpy(audio_data).float().to(self._device)
            audio_tensor = audio_tensor.unsqueeze(0)
            audio = convert_audio(audio_tensor, SAMPLE_RATE, self._denoise_model.sample_rate, self._denoise_model.chin)

            with torch.no_grad():
                denoised = self._denoise_model(audio)[0]

            denoised = denoised * self._voice_boost
            audio_tensor = audio_tensor + denoised
            audio_tensor = audio_tensor / audio_tensor.abs().max()
            audio_tensor = audio_tensor.squeeze(0)
            audio_tensor = audio_tensor.cpu()

            return audio_tensor.numpy()
        except Exception as e:
            logger.error(f"Error in _boost_speech: {e}", exc_info=True)
            return audio_data

    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect voice activity in the audio chunk."""
        try:
            audio_tensor = torch.from_numpy(audio_chunk).float().to(self._device)
            # Ensure input is in the right format for silero_vad
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
                
            # Use the Silero VAD model to get speech timestamps
            timestamps = silero_vad.get_speech_timestamps(
                audio=audio_tensor,
                model=self._vad_model,
                threshold=self._conditioning.vad_threshold,
                sampling_rate=SAMPLE_RATE
            )
            
            return len(timestamps) > 0
        except Exception as e:
            logger.error(f"Error in _detect_voice_activity: {e}", exc_info=True)
            return False

    def _transcribe(self, audio_data: np.ndarray) -> List[Word]:
        """Transcribe audio data to text with word timestamps."""
        try:
            if self._language is not None:
                segments, info = self._model.transcribe(
                    audio_data, 
                    beam_size=5, 
                    language=self._language, 
                    condition_on_previous_text=False, 
                    word_timestamps=True
                )
            else:
                segments, info = self._model.transcribe(
                    audio_data, 
                    beam_size=5, 
                    condition_on_previous_text=False, 
                    word_timestamps=True
                )
            
            transcription = []
            for segment in segments:
                for word in segment.words:
                    transcription.append(Word(text=word.word, start=word.start, end=word.end))
            return transcription
        except Exception as e:
            logger.error(f"Error in _transcribe: {e}", exc_info=True)
            return []

    def _generate_speaker_embedding(self, audio_data: torch.FloatTensor, start: float, end: float) -> torch.FloatTensor:
        """Generate speaker embeddings from audio segment."""
        try:
            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            audio_data = self._split_audio_by_timestamps(audio_data=audio_data, start=start, end=end)

            # Ensure the audio segment is long enough (at least 1 second)
            min_length = SAMPLE_RATE  # 1 second at 16000 Hz
            if audio_data.shape[1] < min_length:
                # Pad the audio data if it's too short
                padding = torch.zeros(1, min_length - audio_data.shape[1], device=audio_data.device)
                audio_data = torch.cat([audio_data, padding], dim=1)

            return self._speaker_embedding_model.encode_batch(audio_data)
        except Exception as e:
            logger.error(f"Error generating speaker embedding: {str(e)}")
            return torch.zeros(1, 512, device=self._device)  # Return a zero embedding as a fallback
    
    def start_sync(self) -> Generator[ContextDatapoint, None, None]:
        """
        Synchronous generator for live voice analysis using silence detection.
        Yields transcribed sentences after a period of silence follows speech.
        """
        # Ensure thread exists and is properly initialized
        logger.info("Starting synchronous transcription...")
        
        # Start recording thread if not already running
        if not self._recording_thread.is_alive():
            try:
                # Reset recording flag to ensure it's running
                self._is_recording = True
                
                # Clear the queue to prevent processing old data
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Start the recording thread
                logger.info(f"Starting recording thread with mic index {self._microphone_index}")
                self._recording_thread.start()
                
                # Wait briefly to ensure thread is running
                time.sleep(0.5)
                
                if not self._recording_thread.is_alive():
                    logger.error("Recording thread failed to start")
                    raise RuntimeError("Failed to start recording thread")
                    
                logger.info("Recording thread started successfully")
            except Exception as e:
                logger.error(f"Error starting recording thread: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to start transcription: {str(e)}")
        else:
            logger.info("Recording thread is already running")

        # Processing variables
        current_audio_data = None
        first_audio_chunk = None
        silence_counter = 0
        continuous_silence_limit = self._max_silence_chunks
        
        last_speech_time = None
        MIN_SILENCE_DURATION = 1.5  # seconds to consider speech finished
        
        logger.info("Starting transcription processing loop...")

        # Main processing loop
        while self._is_recording:
            try:
                # Get audio from queue with timeout
                try:
                    audio_chunk = self._audio_queue.get(timeout=0.1)
                    
                    # Process the audio chunk (boost and VAD)
                    audio_chunk = self._boost_speech(audio_chunk)
                    speech_detected = self._detect_voice_activity(audio_chunk)
                    
                    # Process based on speech detection
                    if speech_detected:
                        logger.debug("Speech detected in audio chunk")
                        last_speech_time = time.time()
                        silence_counter = 0
                        
                        # Initialize or append to audio data
                        if current_audio_data is None:
                            if first_audio_chunk is not None:
                                current_audio_data = np.concatenate((first_audio_chunk, audio_chunk))
                            else:
                                current_audio_data = audio_chunk
                        else:
                            current_audio_data = np.concatenate((current_audio_data, audio_chunk))
                    else:
                        # No speech detected
                        if current_audio_data is not None:
                            # We had speech before, now silent - add to buffer and check silence duration
                            current_audio_data = np.concatenate((current_audio_data, audio_chunk))
                            silence_counter += 1
                            
                            # Check if we have enough silence to consider speech finished
                            if silence_counter >= continuous_silence_limit or (last_speech_time and time.time() - last_speech_time > MIN_SILENCE_DURATION):
                                # Process the accumulated audio and generate transcription
                                if len(current_audio_data) > SAMPLE_RATE * 0.5:  # At least 0.5 seconds of audio
                                    logger.info("Processing completed utterance")
                                    transcription = self._transcribe(current_audio_data)
                                    
                                    if transcription:
                                        # Generate speaker embeddings
                                        audio_tensor = torch.from_numpy(current_audio_data).float().to(self._device)
                                        for i, word in enumerate(transcription):
                                            transcription[i].speaker_embedding = self._generate_speaker_embedding(
                                                audio_tensor, transcription[i].start, transcription[i].end
                                            )
                                        
                                        # Construct datapoint
                                        voice = self.resolve_speaker(words=transcription)
                                        datapoint = ContextDatapoint(
                                            source=ContextSource_Voice(speaker=voice),
                                            content=VoiceProcessingHelpers.word_array_to_string(transcription)
                                        )
                                        
                                        logger.info(f"Yielding transcription: '{datapoint.content}'")
                                        
                                        # Reset state for next utterance
                                        current_audio_data = None
                                        first_audio_chunk = None
                                        silence_counter = 0
                                        last_speech_time = None
                                        
                                        # Yield the result
                                        yield datapoint
                                else:
                                    # Too short, reset
                                    logger.debug("Discarding audio segment (too short)")
                                    current_audio_data = None
                                    first_audio_chunk = None
                                    silence_counter = 0
                                    last_speech_time = None
                        else:
                            # No previous speech, store this chunk in case it's the beginning of speech
                            first_audio_chunk = audio_chunk
                
                except queue.Empty:
                    # Queue is empty, just continue loop
                    pass
                    
            except Exception as e:
                logger.error(f"Error in transcription processing: {str(e)}", exc_info=True)
                time.sleep(0.5)  # Sleep briefly before retrying

        logger.info("Transcription processing loop ended (_is_recording became False)")
    
    # For compatibility with the API
    def start(self) -> Generator[ContextDatapoint, None, None]:
        """Forward to start_sync for API compatibility."""
        logger.info("start() called, forwarding to start_sync()")
        return self.start_sync()
        
    def resolve_speaker(self, words: List[Word]) -> str:
        """
        Finds the name of the current speaker 
        """
        try:
            embedding_list = []
            for word in words:
                embedding_list.append(word.speaker_embedding)

            avg_embedding = VoiceProcessingHelpers.take_average_embedding(embedding_list)

            self._voice_database_manager.open()

            voice = self._voice_database_manager.get_voice_name_from_embedding(avg_embedding)

            if voice and voice[1] > 0.8: # If voice was found and it's close enough use it. Otherwise create a new one
                voice_name = voice[0]
            else:
                voice_name = self._voice_database_manager.create_unknown_voice(avg_embedding)

            self._voice_database_manager.close()

            return voice_name
        except Exception as e:
            logger.error(f"Error in resolve_speaker: {str(e)}", exc_info=True)
            return "user"  # Default fallback
        
    def close(self) -> None:
        """
        Ends the execution of this script. No more data will be yielded by the generator.
        """
        # Set flag first to signal threads to stop
        self._is_recording = False
        
        # Clear the queue to prevent blocking
        try:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error clearing audio queue: {str(e)}")
            
        # Join the recording thread with timeout
        if hasattr(self, '_recording_thread') and self._recording_thread and self._recording_thread.is_alive():
            try:
                self._recording_thread.join(timeout=2.0)
                if self._recording_thread.is_alive():
                    logger.error("Warning: Recording thread did not terminate within timeout")
            except Exception as e:
                logger.error(f"Error joining recording thread: {str(e)}")

    def _split_audio_by_timestamps(self, audio_data: torch.FloatTensor, start: float, end: float) -> torch.FloatTensor:
        """Split audio data by timestamps."""
        try:
            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)

            # Ensure the indices are within bounds
            if audio_data.ndim == 1:
                end_sample = min(end_sample, audio_data.shape[0])
                start_sample = min(start_sample, end_sample)
                return audio_data[start_sample:end_sample].unsqueeze(0)
            else:
                end_sample = min(end_sample, audio_data.shape[1])
                start_sample = min(start_sample, end_sample)
                return audio_data[:, start_sample:end_sample]
        except Exception as e:
            logger.error(f"Error in _split_audio_by_timestamps: {str(e)}", exc_info=True)
            # Return a small segment of zeros as fallback
            return torch.zeros(1, SAMPLE_RATE, device=audio_data.device)
    
    def _log(self, text: str) -> None:
        """Log messages using the standard logging module."""
        if self._verbose:
            logger.debug(f"[VoiceAnalysis] {text}")
        else:
            logger.debug(f"[VoiceAnalysis] {text}")

class VoiceProcessingHelpers:
    def __init__(self):
        """
        A collection of static methods that act as helpers.
        """
        pass
    
    @staticmethod
    def word_array_to_string(word_array: List[Word]) -> str:
        """
        Extracts the text from an array of word objects and returns it as a string.

        Arguments:
            word_array (List[Word]): The array of word objects that will be converted to a string.

        Returns:
            str: The full extracted text.
        """
        text = ""
        for word in word_array:
            text += word.text
        return text
        
    @staticmethod
    def compare_embeddings(emb1: torch.FloatTensor, emb2: torch.FloatTensor) -> float:
        """
        Compare how "close" two embeddings are to each other.

        Arguments:
            emb1 (torch.FloatTensor): The first embedding to compare.
            emb2 (torch.FloatTensor): The second embedding to compare.

        Returns:
            float: How "close" the embeddings are. Ranges from -1 to 1. Higher is closer.
        """
        return (F.cosine_similarity(emb1.squeeze(), emb2.squeeze(), dim=0).mean().item())
    
    @staticmethod
    def take_average_embedding(embeddings: List[torch.FloatTensor]) -> torch.FloatTensor:
        """
        Takes the average of a List of embeddings.

        Arguments:
            embeddings (List[torch.FloatTensor]): The embedding List.

        Returns:
            torch.FloatTensor: The average embedding.
        """
        return torch.mean(torch.stack(embeddings), dim=0)