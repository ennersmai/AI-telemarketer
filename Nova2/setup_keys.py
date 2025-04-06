import sys
from pathlib import Path

# Add Nova2 directory to the Python path
sys.path.append(str(Path().absolute() / "Nova2"))

# Import only what we need for setting API keys
from nova import *

# Initialize Nova
nova = Nova()

# Set the Groq API key
nova.edit_secret(Secrets.GROQ_API, "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")
print("Groq API key set successfully!")

# Set the ElevenLabs API key
nova.edit_secret(Secrets.ELEVENLABS_API, "sk_27339eb30a8525abc0d83e82ddeb2cd72322e309b0d3fb92")
print("ElevenLabs API key set successfully!")

print("\nAPI keys have been securely stored in Nova's encrypted database.")
print("You can now use them in your application without having to set them again.") 