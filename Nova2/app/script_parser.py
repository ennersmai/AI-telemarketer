"""
Utility for parsing the telemarketer script file.
"""

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'scripts', 'telemarketer_script.md')

def load_script_section(state_name: str, script_path: str = DEFAULT_SCRIPT_PATH) -> Optional[str]:
    """
    Parses the structured Markdown script file to find the dialogue for a specific state.

    Args:
        state_name: The exact name of the state to find (e.g., "QUALIFY_A").
        script_path: The path to the script file. Defaults to DEFAULT_SCRIPT_PATH.

    Returns:
        The dialogue string associated with the state, or None if the state
        or its dialogue is not found.
    """
    logger.debug(f"Attempting to load dialogue for state '{state_name}' from '{script_path}'")
    
    if not os.path.exists(script_path):
        logger.error(f"Script file not found at path: {script_path}")
        return None

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            in_target_state = False
            dialogue_lines: List[str] = []
            state_marker = f"# STATE: {state_name}"
            dialogue_prefix = "Dialogue: "

            for line in f:
                stripped_line = line.strip()

                if in_target_state:
                    # Check if we've hit the next state or end of useful info for this state
                    if stripped_line.startswith("# STATE:") or stripped_line == "---": # Stop if next state or separator found
                        break 
                    
                    # Check for dialogue line(s)
                    if stripped_line.startswith(dialogue_prefix):
                        dialogue_content = stripped_line[len(dialogue_prefix):].strip()
                        if dialogue_content: # Avoid adding empty lines if prefix exists but content doesn't
                             dialogue_lines.append(dialogue_content)
                        # Continue checking subsequent lines in case dialogue spans multiple lines implicitly
                        # (Current script doesn't seem to, but allows for future flexibility)
                        
                elif stripped_line == state_marker:
                    in_target_state = True
                    logger.debug(f"Found state marker: {state_marker}")

            if dialogue_lines:
                full_dialogue = " ".join(dialogue_lines) # Join potential multi-lines
                logger.debug(f"Successfully loaded dialogue for state '{state_name}': '{full_dialogue[:50]}...'")
                return full_dialogue
            else:
                if in_target_state:
                     logger.warning(f"State '{state_name}' found in script, but no subsequent 'Dialogue: ' line was found.")
                else:
                     logger.warning(f"State marker '{state_marker}' not found in script: {script_path}")
                return None

    except FileNotFoundError:
        logger.error(f"Script file not found error during read: {script_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading or parsing script file {script_path}: {e}", exc_info=True)
        return None

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Assume the script is in the expected location relative to this file
    script_file_path = DEFAULT_SCRIPT_PATH 
    
    print(f"Using script path: {script_file_path}")
    
    test_state_1 = "QUALIFY_A"
    dialogue_1 = load_script_section(test_state_1, script_file_path)
    print(f"Dialogue for '{test_state_1}': {dialogue_1}")

    test_state_2 = "GATEKEEPER_RESIST_1"
    dialogue_2 = load_script_section(test_state_2, script_file_path)
    print(f"Dialogue for '{test_state_2}': {dialogue_2}")

    test_state_3 = "NON_EXISTENT_STATE"
    dialogue_3 = load_script_section(test_state_3, script_file_path)
    print(f"Dialogue for '{test_state_3}': {dialogue_3}")
    
    test_state_4 = "INITIALIZING" # State with no dialogue
    dialogue_4 = load_script_section(test_state_4, script_file_path)
    print(f"Dialogue for '{test_state_4}': {dialogue_4}") 