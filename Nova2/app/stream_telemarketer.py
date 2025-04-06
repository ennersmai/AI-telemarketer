"""
Script Manager for AI Telemarketer

This module provides utility functions for parsing scripts and managing
conversation flow based on LLM responses and predefined script states.
"""

import re
from typing import Optional, Tuple, Dict

# Assuming ScriptState and ScriptAction are defined elsewhere (e.g., call_state_manager.py)
# We might need to import them if they aren't globally available or passed in.
# For now, assume they are defined/accessible where needed.
from .call_state_manager import ScriptState, ScriptAction # Add this import

class ScriptManager: # Renamed from StreamingTelemarketer
    """
    Manages script parsing, state transitions, and LLM response parsing.
    """
    
    def __init__(self): # Simplified __init__
        """
        Initialize the Script Manager.
        Loads script definitions.
        """
        self.script_data: Dict[ScriptState, Dict] = self._load_scripts()
        
    def _load_scripts(self) -> Dict[ScriptState, Dict]:
        """
        Loads script definitions.
        Returns a dictionary mapping ScriptState enum members to their data.
        """
        # Maps ScriptState members to their data
        script_map = {
            ScriptState.START: {
                "script": "Hello! This is Isaac from proactiv calling about plastic card solutions. Is this a good time to briefly chat?", 
                "action": ScriptAction.TALK, 
                "next_states": ["CONTINUE_INTRO", "CALLBACK", "DECLINE"]
            },
            ScriptState.CONTINUE_INTRO: {
                "script": "Great. We help businesses like yours enhance customer loyalty and branding with custom plastic cards - things like loyalty cards, gift cards, even MOT reminders. Have you considered using these before?", 
                "action": ScriptAction.LISTEN, 
                "next_states": ["GATHER_INFO_YES", "GATHER_INFO_NO", "NOT_INTERESTED"]
            },
            ScriptState.GATHER_INFO_YES: {
                 "script": "That's good to hear. Could you tell me a bit about how you use them currently?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["QUALIFY_FURTHER", "BOOK_APPOINTMENT", "NOT_INTERESTED"]
            },
            ScriptState.GATHER_INFO_NO: {
                 "script": "Okay, no problem. Many businesses find they're a great way to increase repeat business and make a professional impression. What type of business are you in?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["QUALIFY_FURTHER", "BOOK_APPOINTMENT", "NOT_INTERESTED"]
            },
            ScriptState.QUALIFY_FURTHER: {
                 "script": "Thanks for sharing. Based on that, I think our solutions could really benefit you. Would you be open to a quick 10-15 minute virtual demo next week to see how it works?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["BOOK_APPOINTMENT", "HANDLE_OBJECTION", "NOT_INTERESTED"]
            },
            ScriptState.BOOK_APPOINTMENT: {
                 "script": "Fantastic! What day and time works best for you for a brief online demo?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["CONFIRM_APPOINTMENT", "CALLBACK", "NOT_INTERESTED"]
            },
            ScriptState.CONFIRM_APPOINTMENT: {
                 "script": "Okay, I have you down for [Appointment Time]. I'll send a calendar invite shortly. Thanks for your time, and have a great day!", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.HANDLE_OBJECTION: {
                 "script": "I understand. [Address Objection based on user input]. Would a different time work, perhaps?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["BOOK_APPOINTMENT", "NOT_INTERESTED", "CALLBACK"]
            },
            ScriptState.NOT_INTERESTED: {
                 "script": "Okay, I understand. Thanks for your time today. Have a great day.", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.CALLBACK: {
                 "script": "Sure, when would be a better time to reach you?", 
                 "action": ScriptAction.LISTEN, 
                 "next_states": ["SCHEDULE_CALLBACK", "NOT_INTERESTED"]
            },
            ScriptState.SCHEDULE_CALLBACK: {
                 "script": "Got it, I'll call you back then. Thanks!", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.DECLINE: {
                 "script": "No problem at all. Thanks for letting me know. Have a good day.", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.FAREWELL: {
                 "script": "Thanks again, goodbye!", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.HANGUP: {
                 "script": "Okay, goodbye.", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
            },
            ScriptState.ERROR_STATE: {
                 "script": "I seem to be having some technical difficulties. Apologies for that. Goodbye.", 
                 "action": ScriptAction.HANGUP, 
                 "next_states": []
             }
             # TODO: Add entries for ALL ScriptState members defined in call_state_manager.py
             # Missing states like GREETING, GATEKEEPER_DETECTED etc. will cause errors later if accessed.
        }
        # Basic validation (optional): Check if all ScriptState members have an entry
        for state in ScriptState:
            if state not in script_map and state != ScriptState.INITIALIZING: # Allow INITIALIZING to not have script data
                 # warnings.warn(f"Script data missing for state: {state.value}")
                 # Add dummy data to prevent KeyErrors later? Or handle missing keys gracefully.
                 script_map[state] = {"script": f"(Missing script for {state.value})", "action": ScriptAction.HANGUP, "next_states": []}
                 
        return script_map

    # --- Renamed/Modified script access methods --- 
    def get_script_data(self, state: ScriptState) -> Optional[Dict]:
        """Retrieves the data dictionary for a given ScriptState member."""
        return self.script_data.get(state)

    def get_script_text(self, state: ScriptState) -> Optional[str]:
        """
        Returns the dialogue script text for the given state.
        """
        data = self.get_script_data(state)
        return data.get("script") if data else None
        
    def get_script_action(self, state: ScriptState) -> Optional[ScriptAction]:
        """
        Returns the ScriptAction for the given state.
        """
        data = self.get_script_data(state)
        return data.get("action") if data else None
        
    # --- Kept parsing logic --- 
    def parse_llm_response(self, response_content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the LLM response to extract dialogue and the suggested next state name (string).
        Expects the next state in the format: <next_state>STATE_NAME</next_state>
        """
        dialogue_text = response_content
        suggested_next_state_str = None
        
        match = re.search(r"<next_state>(.*?)</next_state>\s*$", response_content, re.IGNORECASE | re.DOTALL)
        
        if match:
            suggested_next_state_str = match.group(1).strip().upper() # Standardize to uppercase
            dialogue_text = response_content[:match.start()].strip()
            
        return dialogue_text, suggested_next_state_str

    # --- Modified validation logic --- 
    def validate_and_get_next_state(self, current_state: ScriptState, suggested_state_name: Optional[str]) -> Optional[ScriptState]:
        """
        Validates if the suggested state transition is allowed from the current state 
        and returns the corresponding ScriptState enum member if valid.
        """
        if not suggested_state_name:
            return None 
            
        current_data = self.get_script_data(current_state)
        if not current_data:
             # Current state has no defined script data
             return None
             
        # Check if the suggested state name (string) is in the list of allowed next state strings
        if suggested_state_name in current_data.get("next_states", []):
            try:
                # Convert the valid next state name string back to a ScriptState enum member
                return ScriptState[suggested_state_name]
            except KeyError:
                # The suggested_state_name exists in next_states list but not in ScriptState enum itself
                return None 
        else:
            # Invalid transition
            return None

# --- Removed Example Usage Block --- 