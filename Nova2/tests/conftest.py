import sys
import os

# Add the project root directory (Nova2) to the Python path
# This allows tests to import modules from 'app' directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Added to sys.path in conftest.py: {project_root}") # Optional: for debugging
