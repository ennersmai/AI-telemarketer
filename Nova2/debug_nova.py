from nova import *
import time
import os

print("Initializing Nova...")
nova = Nova()

print("Configuring TTS with ElevenLabs...")
# Set up the inference engine (ElevenLabs)
inference_engine = InferenceEngineElevenlabs()

# Configure with a default voice
conditioning = TTSConditioning(
    model="eleven_turbo_v2",
    voice="21m00Tcm4TlvDq8ikWAM",    # Rachel voice ID
    expressivness=0.3,
    stability=0.5,
    similarity_boost=0.75,
    use_speaker_boost=True
)

# Apply the configuration
nova.configure_tts(inference_engine=inference_engine, conditioning=conditioning)
print("Loading TTS model...")
nova.apply_config_tts()
print("TTS model loaded successfully!")

# Check the nova structure
print("\nDebug Nova Structure:")
print(f"hasattr(nova, '_tts_manager'): {hasattr(nova, '_tts_manager')}")

if hasattr(nova, '_tts_manager'):
    tts_manager = getattr(nova, '_tts_manager')
    print(f"Type of nova._tts_manager: {type(tts_manager)}")
    print(f"Dir of nova._tts_manager: {dir(tts_manager)}")
    
    # Check inference engine
    print(f"\nhasattr(tts_manager, '_inference_engine'): {hasattr(tts_manager, '_inference_engine')}")
    
    if hasattr(tts_manager, '_inference_engine'):
        inference_engine = tts_manager._inference_engine
        print(f"Type of inference_engine: {type(inference_engine)}")
        print(f"Dir of inference_engine: {dir(inference_engine)}")
        
        # Check streaming capability
        if hasattr(inference_engine, 'run_inference'):
            run_inference_sig = str(inspect.signature(inference_engine.run_inference))
            print(f"\nSignature of inference_engine.run_inference: {run_inference_sig}")
    else:
        print("No inference engine found in tts_manager")
else:
    print("No TTS manager found in nova")

# Try to run TTS
print("\nTrying to run TTS...")
test_text = "Hello, this is a test."
start_time = time.time()
try:
    audio = nova.run_tts(test_text)
    end_time = time.time()
    print(f"TTS ran successfully in {end_time - start_time:.2f} seconds.")
    print(f"Type of audio: {type(audio)}")
    print(f"Dir of audio: {dir(audio)}")
except Exception as e:
    print(f"Error running TTS: {e}")

print("\nDebug complete.") 