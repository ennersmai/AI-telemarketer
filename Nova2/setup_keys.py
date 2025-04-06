import sys
from pathlib import Path

# Add Nova2 directory to the Python path
sys.path.append(str(Path().absolute() / "Nova2"))

# Import only what we need for setting API keys
from nova import *

# Initialize Nova
nova = Nova()

print("\nAPI keys have been securely stored in Nova's encrypted database.")
print("You can now use them in your application without having to set them again.") 
