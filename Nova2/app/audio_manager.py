"""
Description: This script is responsible for audio playback.
"""

import threading
import io

from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import wave

class AudioData:
    def __init__(self) -> None:
        self._audio_data = None

    def _store_audio(self, data: list[bytes]) -> None:
        data_full = b''.join(data)

        self._audio_data = self._process_audio(data_full)

    def _process_audio(self, data: bytes) -> AudioSegment:
        if data.startswith(b'RIFF'): # Handle wave audio
            with io.BytesIO(data) as bio:
                with wave.open(bio, 'rb') as wave_file:
                    sample_width = wave_file.getsampwidth()
                    channels = wave_file.getnchannels()
                    framerate = wave_file.getframerate()
                        
                    audio_data = wave_file.readframes(wave_file.getnframes())
                        
            return AudioSegment(
                data=audio_data,
                sample_width=sample_width,
                frame_rate=framerate,
                channels=channels
            )
        else: # Handle mp3 audio
            return AudioSegment.from_file(
                io.BytesIO(data),
                format='mp3'
            )
        
class AudioPlayer:
    def __init__(self) -> None:
        """
        The audio player. Playback can be interrupted.
        """
        self._current_playback = None # Initialize to None
        self._stop_event = threading.Event()
        self._player_thread = None # Initialize thread attribute

    def play_audio(self, audio_data: AudioData) -> None:
        # Ensure previous playback is stopped/cleaned up if necessary
        if self.is_playing():
            self.stop()
        
        self._stop_event.clear() # Reset stop event for new playback
        self._player_thread = threading.Thread(target=self._player, daemon=True, args=(audio_data,))
        self._player_thread.start()

    def _player(self, audio_data: AudioData) -> None:
        """
        Plays the audio data in a separate thread.
        """
        playback_obj = None # Local variable for playback object
        try:
            playback_obj = _play_with_simpleaudio(audio_data._audio_data)
            self._current_playback = playback_obj # Assign object to instance variable *after* creation
                    
            # Wait for playback to finish or stop event
            # Check if playback_obj exists and has is_playing method
            while playback_obj and hasattr(playback_obj, 'is_playing') and playback_obj.is_playing() and not self._stop_event.is_set():
                threading.Event().wait(0.1) # Use threading.Event().wait for a cleaner sleep
                        
            # Stop playback if still playing (and stop wasn't requested)
            if playback_obj and hasattr(playback_obj, 'is_playing') and playback_obj.is_playing() and not self._stop_event.is_set():
                 if hasattr(playback_obj, 'stop'):
                     playback_obj.stop()
                
        except Exception as e:
             # Log any exceptions during playback
             print(f"[ERROR][AudioPlayer._player] Exception during playback: {e}") # Added basic logging
        finally:
            # Ensure cleanup happens
            if self._current_playback == playback_obj: # Only clear if it's the *same* playback object
                self._current_playback = None

    def stop(self) -> None:
        """
        Interrupt the current playback.
        """
        self._stop_event.set() # Signal the player loop to stop
        
        # Check if playback object exists and try stopping it
        playback_obj = self._current_playback
        if playback_obj and hasattr(playback_obj, 'is_playing') and hasattr(playback_obj, 'stop'):
            try:
                if playback_obj.is_playing():
                    playback_obj.stop()
            except Exception as e:
                 print(f"[ERROR][AudioPlayer.stop] Exception stopping playback object: {e}")


        # Wait for the thread to finish
        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=1.0) # Add a timeout

        # Reset state
        self._player_thread = None
        self._current_playback = None
        # It's generally good practice to clear the event after stopping, 
        # although play_audio already does this before starting.
        # self._stop_event.clear() 

    def is_playing(self) -> bool:
        """ Checks if audio is currently playing. """
        playback_obj = self._current_playback
        # Check if the object exists, has the method, and the method returns True
        return (playback_obj is not None and 
                hasattr(playback_obj, 'is_playing') and 
                callable(playback_obj.is_playing) and 
                playback_obj.is_playing())