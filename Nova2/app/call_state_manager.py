"""
Call State Manager for AI Telemarketer

This module provides a robust state management system for tracking
conversation states and supporting persistence across calls.
"""

import json
import time
import asyncio
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
import sqlite3
import aiosqlite

# Configure logging
logger = logging.getLogger("call_state_manager")

# --- Define NEW conversation states based on script ---
class ScriptState(str, Enum):
    INITIALIZING = "INITIALIZING"
    GREETING = "GREETING"
    START = "START"
    CONTINUE_INTRO = "CONTINUE_INTRO"
    GATEKEEPER_DETECTED = "GATEKEEPER_DETECTED"
    GATEKEEPER_INTRO = "GATEKEEPER_INTRO"
    GATEKEEPER_RESIST_1 = "GATEKEEPER_RESIST_1"
    GATEKEEPER_RESIST_2 = "GATEKEEPER_RESIST_2"
    OWNER_CONNECTING = "OWNER_CONNECTING"
    CALL_END_NO_OWNER = "CALL_END_NO_OWNER"
    OWNER_CONFIRMED = "OWNER_CONFIRMED"
    OWNER_INTRO = "OWNER_INTRO"
    OWNER_SMALLTALK = "OWNER_SMALLTALK"
    OWNER_INTEREST = "OWNER_INTEREST"
    OWNER_INTEREST_NEGATIVE = "OWNER_INTEREST_NEGATIVE"
    QUALIFY_START = "QUALIFY_START"
    QUALIFY_A = "QUALIFY_A"
    QUALIFY_B = "QUALIFY_B"
    QUALIFY_C = "QUALIFY_C"
    QUALIFY_D = "QUALIFY_D"
    QUALIFY_E = "QUALIFY_E"
    QUALIFY_F = "QUALIFY_F"
    QUALIFY_G = "QUALIFY_G"
    QUALIFY_CHECK = "QUALIFY_CHECK" # Internal logic state
    GATHER_INFO_YES = "GATHER_INFO_YES"
    GATHER_INFO_NO = "GATHER_INFO_NO"
    QUALIFY_FURTHER = "QUALIFY_FURTHER"
    HANDLE_OBJECTION = "HANDLE_OBJECTION"
    BOOK_APPOINTMENT = "BOOK_APPOINTMENT"
    CONFIRM_APPOINTMENT = "CONFIRM_APPOINTMENT"
    CALLBACK = "CALLBACK"
    SCHEDULE_CALLBACK = "SCHEDULE_CALLBACK"
    NOT_INTERESTED = "NOT_INTERESTED"
    DECLINE = "DECLINE"
    HANGUP = "HANGUP" 
    ERROR_STATE = "ERROR_STATE"
    PROBLEM_START = "PROBLEM_START"
    PROBLEM_A = "PROBLEM_A"
    PROBLEM_B = "PROBLEM_B"
    PROBLEM_C = "PROBLEM_C"
    PROBLEM_D = "PROBLEM_D"
    PROBLEM_E = "PROBLEM_E"
    PROBLEM_F = "PROBLEM_F"
    PROBLEM_G = "PROBLEM_G"
    FACTFIND_START = "FACTFIND_START"
    FACTFIND_A = "FACTFIND_A"
    FACTFIND_B = "FACTFIND_B"
    FACTFIND_C = "FACTFIND_C"
    FACTFIND_D = "FACTFIND_D"
    FACTFIND_E = "FACTFIND_E"
    FACTFIND_F = "FACTFIND_F"
    FACTFIND_G = "FACTFIND_G"
    FACTFIND_CARDS = "FACTFIND_CARDS"
    EXPLAIN_STORY_PLASTIC = "EXPLAIN_STORY_PLASTIC"
    EXPLAIN_STORY_KEYFOB = "EXPLAIN_STORY_KEYFOB"
    PRE_CLOSE = "PRE_CLOSE"
    BENEFITS_INTRO = "BENEFITS_INTRO"
    BENEFIT_STORY_A = "BENEFIT_STORY_A"
    BENEFIT_STORY_B = "BENEFIT_STORY_B"
    BENEFIT_STORY_C = "BENEFIT_STORY_C"
    BENEFIT_STORY_D = "BENEFIT_STORY_D"
    BENEFIT_STORY_E = "BENEFIT_STORY_E"
    BENEFIT_STORY_F = "BENEFIT_STORY_F"
    BENEFIT_STORY_G = "BENEFIT_STORY_G"
    EXPLAIN_DEMO = "EXPLAIN_DEMO"
    BOOKING_CLOSE_TIME = "BOOKING_CLOSE_TIME"
    BOOKING_CLOSE_CONFIRM = "BOOKING_CLOSE_CONFIRM"
    CONSOLIDATE_DECISIONMAKERS = "CONSOLIDATE_DECISIONMAKERS"
    CONSOLIDATE_DETAILS = "CONSOLIDATE_DETAILS"
    CONSOLIDATE_NEXTSTAGE = "CONSOLIDATE_NEXTSTAGE"
    FAREWELL = "FAREWELL"
    COMPLETED = "COMPLETED" # Final success state
    ERROR_HANDLE = "ERROR_HANDLE" # State for errors
    CALL_ENDED_ABRUPTLY = "CALL_ENDED_ABRUPTLY" # Final state for disconnects etc.

# --- ADD ScriptAction Enum --- 
class ScriptAction(str, Enum):
    TALK = "TALK"     # AI should speak
    LISTEN = "LISTEN" # AI should wait for user input
    HANGUP = "HANGUP" # AI should end the call
    # Add other actions if needed later (e.g., TRANSFER, PLAY_AUDIO)
# --- End ADD --- 

# (Keep ConversationEvent for now, might be useful for logging or specific triggers)
class ConversationEvent(str, Enum):
    CALL_CONNECTED = "call_connected"
    USER_SPEECH_DETECTED = "user_speech_detected" # Replaces USER_RESPONSE
    LLM_RESPONSE_GENERATED = "llm_response_generated"
    LLM_STATE_SUGGESTED = "llm_state_suggested"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    STATE_TRANSITION_VALID = "state_transition_valid"
    STATE_TRANSITION_INVALID = "state_transition_invalid"
    APPOINTMENT_BOOKED = "appointment_booked"
    END_CALL_REQUESTED = "end_call_requested"
    CALL_TERMINATED = "call_terminated"
    ERROR_OCCURRED = "error_occurred"

# --- Define Valid State Transitions --- 
# Define common exit states allowable from most active states
_COMMON_EXIT_STATES = {ScriptState.FAREWELL, ScriptState.CALL_ENDED_ABRUPTLY, ScriptState.ERROR_HANDLE}

_VALID_TRANSITIONS: Dict[ScriptState, Set[ScriptState]] = {
    ScriptState.INITIALIZING: {ScriptState.GREETING},
    ScriptState.GREETING: {ScriptState.GATEKEEPER_DETECTED, ScriptState.OWNER_CONFIRMED}.union(_COMMON_EXIT_STATES),
    ScriptState.GATEKEEPER_DETECTED: {ScriptState.GATEKEEPER_INTRO, ScriptState.OWNER_CONNECTING}.union(_COMMON_EXIT_STATES),
    ScriptState.GATEKEEPER_INTRO: {ScriptState.OWNER_CONNECTING, ScriptState.GATEKEEPER_RESIST_1}.union(_COMMON_EXIT_STATES),
    ScriptState.GATEKEEPER_RESIST_1: {ScriptState.OWNER_CONNECTING, ScriptState.GATEKEEPER_RESIST_2, ScriptState.CALL_END_NO_OWNER}.union(_COMMON_EXIT_STATES),
    ScriptState.GATEKEEPER_RESIST_2: {ScriptState.OWNER_CONNECTING, ScriptState.CALL_END_NO_OWNER}.union(_COMMON_EXIT_STATES),
    ScriptState.OWNER_CONNECTING: {ScriptState.OWNER_INTRO}.union(_COMMON_EXIT_STATES),
    ScriptState.CALL_END_NO_OWNER: {ScriptState.COMPLETED}, # End state
    ScriptState.OWNER_CONFIRMED: {ScriptState.OWNER_INTRO}.union(_COMMON_EXIT_STATES),
    ScriptState.CONTINUE_INTRO: {
        ScriptState.OWNER_INTEREST, 
        ScriptState.QUALIFY_START, 
        ScriptState.NOT_INTERESTED, 
        ScriptState.DECLINE, 
        ScriptState.CALLBACK
        }.union(_COMMON_EXIT_STATES),
    ScriptState.OWNER_INTRO: {ScriptState.OWNER_SMALLTALK}.union(_COMMON_EXIT_STATES),
    ScriptState.OWNER_SMALLTALK: {ScriptState.OWNER_INTEREST}.union(_COMMON_EXIT_STATES),
    ScriptState.OWNER_INTEREST: {ScriptState.OWNER_INTEREST_NEGATIVE, ScriptState.QUALIFY_START}.union(_COMMON_EXIT_STATES),
    ScriptState.OWNER_INTEREST_NEGATIVE: {ScriptState.QUALIFY_START}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_START: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_B, ScriptState.QUALIFY_C, ScriptState.QUALIFY_D, ScriptState.QUALIFY_E, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G}.union(_COMMON_EXIT_STATES),
    # Qualify states can loop back or go to check
    ScriptState.QUALIFY_A: {ScriptState.QUALIFY_B, ScriptState.QUALIFY_C, ScriptState.QUALIFY_D, ScriptState.QUALIFY_E, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_B: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_C, ScriptState.QUALIFY_D, ScriptState.QUALIFY_E, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_C: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_B, ScriptState.QUALIFY_D, ScriptState.QUALIFY_E, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_D: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_B, ScriptState.QUALIFY_C, ScriptState.QUALIFY_E, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_E: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_B, ScriptState.QUALIFY_C, ScriptState.QUALIFY_D, ScriptState.QUALIFY_F, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_F: {ScriptState.QUALIFY_A, ScriptState.QUALIFY_B, ScriptState.QUALIFY_C, ScriptState.QUALIFY_D, ScriptState.QUALIFY_E, ScriptState.QUALIFY_G, ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_G: {ScriptState.QUALIFY_CHECK}.union(_COMMON_EXIT_STATES),
    ScriptState.QUALIFY_CHECK: {ScriptState.PROBLEM_START, ScriptState.EXPLAIN_STORY_PLASTIC}, # Internal state, app should handle transition
    ScriptState.PROBLEM_START: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_B, ScriptState.PROBLEM_C, ScriptState.PROBLEM_D, ScriptState.PROBLEM_E, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G}.union(_COMMON_EXIT_STATES),
    # Problem states can loop back or go to factfind
    ScriptState.PROBLEM_A: {ScriptState.PROBLEM_B, ScriptState.PROBLEM_C, ScriptState.PROBLEM_D, ScriptState.PROBLEM_E, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_B: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_C, ScriptState.PROBLEM_D, ScriptState.PROBLEM_E, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_C: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_B, ScriptState.PROBLEM_D, ScriptState.PROBLEM_E, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_D: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_B, ScriptState.PROBLEM_C, ScriptState.PROBLEM_E, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_E: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_B, ScriptState.PROBLEM_C, ScriptState.PROBLEM_D, ScriptState.PROBLEM_F, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_F: {ScriptState.PROBLEM_A, ScriptState.PROBLEM_B, ScriptState.PROBLEM_C, ScriptState.PROBLEM_D, ScriptState.PROBLEM_E, ScriptState.PROBLEM_G, ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.PROBLEM_G: {ScriptState.FACTFIND_START}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_START: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_B, ScriptState.FACTFIND_C, ScriptState.FACTFIND_D, ScriptState.FACTFIND_E, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    # Factfind states can loop back or go to cards
    ScriptState.FACTFIND_A: {ScriptState.FACTFIND_B, ScriptState.FACTFIND_C, ScriptState.FACTFIND_D, ScriptState.FACTFIND_E, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_B: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_C, ScriptState.FACTFIND_D, ScriptState.FACTFIND_E, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_C: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_B, ScriptState.FACTFIND_D, ScriptState.FACTFIND_E, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_D: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_B, ScriptState.FACTFIND_C, ScriptState.FACTFIND_E, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_E: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_B, ScriptState.FACTFIND_C, ScriptState.FACTFIND_D, ScriptState.FACTFIND_F, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_F: {ScriptState.FACTFIND_A, ScriptState.FACTFIND_B, ScriptState.FACTFIND_C, ScriptState.FACTFIND_D, ScriptState.FACTFIND_E, ScriptState.FACTFIND_G, ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_G: {ScriptState.FACTFIND_CARDS}.union(_COMMON_EXIT_STATES),
    ScriptState.FACTFIND_CARDS: {ScriptState.EXPLAIN_STORY_PLASTIC, ScriptState.EXPLAIN_STORY_KEYFOB}.union(_COMMON_EXIT_STATES),
    ScriptState.EXPLAIN_STORY_PLASTIC: {ScriptState.PRE_CLOSE}.union(_COMMON_EXIT_STATES),
    ScriptState.EXPLAIN_STORY_KEYFOB: {ScriptState.PRE_CLOSE, ScriptState.BENEFITS_INTRO}.union(_COMMON_EXIT_STATES),
    ScriptState.PRE_CLOSE: {ScriptState.EXPLAIN_DEMO, ScriptState.BENEFITS_INTRO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFITS_INTRO: {ScriptState.BENEFIT_STORY_A, ScriptState.BENEFIT_STORY_B, ScriptState.BENEFIT_STORY_C, ScriptState.BENEFIT_STORY_D, ScriptState.BENEFIT_STORY_E, ScriptState.BENEFIT_STORY_F, ScriptState.BENEFIT_STORY_G}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_A: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_B: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_C: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_D: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_E: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_F: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.BENEFIT_STORY_G: {ScriptState.EXPLAIN_DEMO}.union(_COMMON_EXIT_STATES),
    ScriptState.EXPLAIN_DEMO: {ScriptState.BOOKING_CLOSE_TIME}.union(_COMMON_EXIT_STATES),
    ScriptState.BOOKING_CLOSE_TIME: {ScriptState.BOOKING_CLOSE_CONFIRM}.union(_COMMON_EXIT_STATES), # Add objection handling later
    ScriptState.BOOKING_CLOSE_CONFIRM: {ScriptState.CONSOLIDATE_DECISIONMAKERS}.union(_COMMON_EXIT_STATES),
    ScriptState.CONSOLIDATE_DECISIONMAKERS: {ScriptState.CONSOLIDATE_DETAILS}.union(_COMMON_EXIT_STATES),
    ScriptState.CONSOLIDATE_DETAILS: {ScriptState.CONSOLIDATE_NEXTSTAGE}.union(_COMMON_EXIT_STATES),
    ScriptState.CONSOLIDATE_NEXTSTAGE: {ScriptState.FAREWELL}.union(_COMMON_EXIT_STATES),
    ScriptState.FAREWELL: {ScriptState.COMPLETED}, # End state
    # Error/End states don't transition further normally
    ScriptState.COMPLETED: set(),
    ScriptState.ERROR_HANDLE: _COMMON_EXIT_STATES, # Allow trying to exit gracefully
    ScriptState.CALL_ENDED_ABRUPTLY: set(),
}

class CallStateMachine:
    """State machine for tracking the state of a conversation in a call"""
    
    def __init__(self, call_sid: str, initial_state: ScriptState = ScriptState.GREETING):
        """Initialize the state machine for a call"""
        self.call_sid = call_sid
        self.state: ScriptState = initial_state
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.history: List[Dict[str, Any]] = [] # Explicit type hint
        # self.user_responses = {} # Likely replace with just history
        self.is_outbound = False
        self.script_name = "telemarketer_script_v1" # Store script name/version used
        self.from_number = ""
        self.to_number = ""
        self.duration = 0
        # self.intents_detected = [] # Less relevant if LLM drives flow
        # self.keywords_detected = [] # Less relevant
        # self.sentiment_scores = [] # Less relevant

        # --- NEW Fields for script data ---
        self.owner_name: Optional[str] = None
        self.gatekeeper_name: Optional[str] = None
        self.agent_name: Optional[str] = None # Consider setting from config
        self.client_name: Optional[str] = None # Consider setting from config
        self.qualifiers_met: Dict[str, bool] = {} # e.g., {"A": True, "F": False}
        self.highlighted_problem: Optional[str] = None # e.g., "A"
        self.appointment_time: Optional[str] = None # Store as ISO string or similar
        self.contact_mobile: Optional[str] = None
        self.contact_email: Optional[str] = None
        self.additional_decision_makers: Optional[str] = None
        # --- End NEW Fields ---
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state machine to a dictionary for storage"""
        return {
            "call_sid": self.call_sid,
            "state": self.state.value, # Store enum value
            "start_time": self.start_time,
            "last_update_time": self.last_update_time,
            "history": json.dumps(self.history), # Store history as JSON string
            # "user_responses": json.dumps(self.user_responses),
            "is_outbound": self.is_outbound,
            "script_name": self.script_name,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "duration": self.duration,
            # "intents_detected": json.dumps(self.intents_detected),
            # "keywords_detected": json.dumps(self.keywords_detected),
            # "sentiment_scores": json.dumps(self.sentiment_scores),
            # --- NEW Fields ---
            "owner_name": self.owner_name,
            "gatekeeper_name": self.gatekeeper_name,
            "agent_name": self.agent_name,
            "client_name": self.client_name,
            "qualifiers_met": json.dumps(self.qualifiers_met), # Store dict as JSON string
            "highlighted_problem": self.highlighted_problem,
            "appointment_time": self.appointment_time,
            "contact_mobile": self.contact_mobile,
            "contact_email": self.contact_email,
            "additional_decision_makers": self.additional_decision_makers,
            # --- End NEW Fields ---
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CallStateMachine':
        """Create a state machine from a dictionary"""
        # Use INITIALIZING as default if state is missing or invalid
        initial_state = ScriptState(data.get("state", ScriptState.INITIALIZING.value))
        state_machine = cls(data["call_sid"], initial_state=initial_state)
        state_machine.start_time = data.get("start_time", time.time())
        state_machine.last_update_time = data.get("last_update_time", time.time())
        state_machine.history = json.loads(data.get("history", "[]"))
        # state_machine.user_responses = json.loads(data.get("user_responses", "{}"))
        state_machine.is_outbound = data.get("is_outbound", False)
        state_machine.script_name = data.get("script_name", "unknown")
        state_machine.from_number = data.get("from_number", "")
        state_machine.to_number = data.get("to_number", "")
        state_machine.duration = data.get("duration", 0)
        # state_machine.intents_detected = json.loads(data.get("intents_detected", "[]"))
        # state_machine.keywords_detected = json.loads(data.get("keywords_detected", "[]"))
        # state_machine.sentiment_scores = json.loads(data.get("sentiment_scores", "[]"))
        # --- NEW Fields ---
        state_machine.owner_name = data.get("owner_name")
        state_machine.gatekeeper_name = data.get("gatekeeper_name")
        state_machine.agent_name = data.get("agent_name")
        state_machine.client_name = data.get("client_name")
        state_machine.qualifiers_met = json.loads(data.get("qualifiers_met", "{}"))
        state_machine.highlighted_problem = data.get("highlighted_problem")
        state_machine.appointment_time = data.get("appointment_time")
        state_machine.contact_mobile = data.get("contact_mobile")
        state_machine.contact_email = data.get("contact_email")
        state_machine.additional_decision_makers = data.get("additional_decision_makers")
        # --- End NEW Fields ---
        return state_machine
    
    def add_history_entry(self, speaker: str, text: str):
        """Add an entry to the conversation history"""
        # Remove intent for simplicity with LLM-driven flow
        entry = {
            "timestamp": time.time(),
            "speaker": speaker, # 'user' or 'assistant'
            "text": text,
        }
        self.history.append(entry)
        self.last_update_time = time.time()
        # Optional: Prune history if it gets too long

    # --- Implement NEW transition logic --- 
    def validate_and_set_state(self, suggested_next_state_str: Optional[str]) -> bool:
        """
        Validates the suggested next state against the defined transition map 
        and updates the internal state if valid.
        Returns True if the state was updated, False otherwise.
        """
        current_state = self.state
        logger.debug(f"Current state: {current_state}. Attempting transition based on suggestion: '{suggested_next_state_str}'")

        if not suggested_next_state_str:
            logger.warning(f"No next state suggested by LLM for call {self.call_sid}. Staying in state {current_state}.")
            # Fire event for logging/monitoring?
            # self.log_event(ConversationEvent.STATE_TRANSITION_INVALID, {"reason": "No suggestion"})
            return False

        try:
            suggested_next_state = ScriptState(suggested_next_state_str)
        except ValueError:
            logger.error(f"Invalid state value '{suggested_next_state_str}' suggested by LLM for call {self.call_sid}. Staying in state {current_state}.")
            # Fire event?
            # self.log_event(ConversationEvent.STATE_TRANSITION_INVALID, {"reason": "Invalid state value", "suggested": suggested_next_state_str})
            return False

        # Get valid transitions for the current state
        valid_next_states = _VALID_TRANSITIONS.get(current_state, set()) # Default to empty set if state not in map (shouldn't happen)

        if suggested_next_state in valid_next_states:
            self.state = suggested_next_state
            self.last_update_time = time.time()
            logger.info(f"State transitioned for call {self.call_sid}: {current_state.value} -> {self.state.value}")
            # Fire event?
            # self.log_event(ConversationEvent.STATE_TRANSITION_VALID, {"old_state": current_state, "new_state": self.state})
            return True
        else:
            logger.warning(f"Invalid transition suggested for call {self.call_sid}: {current_state.value} -> {suggested_next_state.value}. Staying in state {current_state.value}.")
            logger.debug(f"Valid transitions from {current_state.value}: {valid_next_states}")
            # Fire event?
            # self.log_event(ConversationEvent.STATE_TRANSITION_INVALID, {"reason": "Transition not allowed", "current": current_state, "suggested": suggested_next_state})
            return False
        
    # --- NEW Methods to update data --- 
    def set_data_field(self, field_name: str, value: Any):
        """Sets a specific data field if it exists."""
        if hasattr(self, field_name):
            setattr(self, field_name, value)
            self.last_update_time = time.time()
            logger.debug(f"Set {field_name}={value} for call {self.call_sid}")
        else:
            logger.warning(f"Attempted to set non-existent field '{field_name}' on CallStateMachine for {self.call_sid}")

    def set_qualifier(self, qualifier_letter: str, value: bool):
        """Sets a specific qualifier result."""
        if qualifier_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            self.qualifiers_met[qualifier_letter.upper()] = value
            self.last_update_time = time.time()
            logger.debug(f"Set qualifier {qualifier_letter}={value} for call {self.call_sid}")
        else:
            logger.warning(f"Invalid qualifier letter '{qualifier_letter}' provided for call {self.call_sid}")
    # --- End NEW Methods ---


class CallStateManager:
    """Manager for tracking and persisting call states"""
    
    def __init__(self, db_path: str = "telemarketer_calls.db"):
        """Initialize the call state manager"""
        self.db_path = db_path
        self.active_calls: Dict[str, CallStateMachine] = {}
        # self.call_handlers: Dict[str, Callable] = {} # Likely remove if LLM drives flow
        self._db_lock = asyncio.Lock()
        self._setup_database()
        
    def _setup_database(self):
        """Set up the SQLite database for call state persistence"""
        # Use synchronous connection for setup
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create calls table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS calls (
            call_sid TEXT PRIMARY KEY,
            state TEXT,
            start_time REAL,
            last_update_time REAL,
            history TEXT, 
            is_outbound BOOLEAN,
            script_name TEXT,
            from_number TEXT,
            to_number TEXT,
            duration REAL,
            owner_name TEXT,
            gatekeeper_name TEXT,
            agent_name TEXT,
            client_name TEXT,
            qualifiers_met TEXT,
            highlighted_problem TEXT,
            appointment_time TEXT,
            contact_mobile TEXT,
            contact_email TEXT,
            additional_decision_makers TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        # Remove old/unused columns if necessary via ALTER TABLE (handle carefully)
        # Example: Check if column exists before altering
        # cursor.execute("PRAGMA table_info(calls)")
        # columns = [info[1] for info in cursor.fetchall()]
        # if 'user_responses' in columns:
        #     cursor.execute("ALTER TABLE calls DROP COLUMN user_responses") # SQLite specific syntax varies

        # Create table for rate limiting (if still needed)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS number_call_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            call_time TIMESTAMP,
            call_sid TEXT, 
            status TEXT 
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup complete.")

    async def get_call_state(self, call_sid: str) -> Optional[CallStateMachine]:
        """Get the state machine for a call, from memory or DB"""
        if call_sid in self.active_calls:
            return self.active_calls[call_sid]
        
        # If not in memory, try loading from DB
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        # Convert row tuple to dictionary based on column names
                        column_names = [description[0] for description in cursor.description]
                        data = dict(zip(column_names, row))
                        state_machine = CallStateMachine.from_dict(data)
                        self.active_calls[call_sid] = state_machine # Cache in memory
                        logger.info(f"Loaded call state for {call_sid} from DB.")
                        return state_machine
        return None

    async def create_call_state(self, call_sid: str, initial_data: Dict[str, Any]) -> CallStateMachine:
        """Create a new state machine for a call and persist it"""
        if call_sid in self.active_calls:
            logger.warning(f"Attempted to create state for already active call {call_sid}")
            return self.active_calls[call_sid]
            
        state_machine = CallStateMachine(call_sid)
        
        # Apply initial data
        state_machine.is_outbound = initial_data.get('is_outbound', False)
        state_machine.script_name = initial_data.get('script_name', state_machine.script_name)
        state_machine.from_number = initial_data.get('from_number', '')
        state_machine.to_number = initial_data.get('to_number', '')
        state_machine.agent_name = initial_data.get('agent_name') # Pass agent name if known
        state_machine.client_name = initial_data.get('client_name') # Pass client name if known

        self.active_calls[call_sid] = state_machine
        await self.save_call_state(call_sid) # Initial save
        logger.info(f"Created new call state for {call_sid}.")
        return state_machine

    async def save_call_state(self, call_sid: str) -> bool:
        """Persist the current state of a call to the database"""
        if call_sid not in self.active_calls:
            logger.error(f"Attempted to save state for inactive call {call_sid}")
            return False
            
        state_machine = self.active_calls[call_sid]
        state_dict = state_machine.to_dict()
        
        # Prepare SQL statement for UPSERT (INSERT OR REPLACE)
        columns = ", ".join(state_dict.keys())
        placeholders = ", ".join(["?"] * len(state_dict))
        sql = f"INSERT OR REPLACE INTO calls ({columns}) VALUES ({placeholders})"
        values = tuple(state_dict.values())
        
        try:
            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(sql, values)
                    await db.commit()
            logger.debug(f"Saved call state for {call_sid} to DB.")
            return True
        except Exception as e:
            logger.error(f"Failed to save call state for {call_sid}: {e}", exc_info=True)
            return False

    async def end_call(self, call_sid: str, final_state: ScriptState = ScriptState.CALL_ENDED_ABRUPTLY) -> bool:
        """Mark a call as ended and remove from active calls"""
        if call_sid in self.active_calls:
            state_machine = self.active_calls[call_sid]
            state_machine.state = final_state # Set final state
            state_machine.duration = time.time() - state_machine.start_time
            state_machine.last_update_time = time.time()
            
            logger.info(f"Ending call {call_sid} with final state {final_state}. Duration: {state_machine.duration:.2f}s")
            
            # Save final state
            await self.save_call_state(call_sid)
            
            # Remove from active memory
            del self.active_calls[call_sid]
            return True
        else:
            logger.warning(f"Attempted to end already inactive call {call_sid}")
            # Optionally update DB status even if not in active memory
            async with self._db_lock:
                 async with aiosqlite.connect(self.db_path) as db:
                    # Check if exists before updating
                    cursor = await db.execute("SELECT 1 FROM calls WHERE call_sid = ?", (call_sid,))
                    exists = await cursor.fetchone()
                    if exists:
                        await db.execute("UPDATE calls SET state = ?, last_update_time = ? WHERE call_sid = ?", 
                                        (final_state.value, time.time(), call_sid))
                        await db.commit()
                        logger.info(f"Updated final state in DB for inactive call {call_sid}.")
            return False

    # --- Rate Limiting / Number History Methods (Keep as is for now) ---
    async def record_call_attempt(self, phone_number: str, call_sid: str, status: str):
        """Record call attempt time for rate limiting and history."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO number_call_history (phone_number, call_time, call_sid, status) VALUES (?, ?, ?, ?)",
                    (phone_number, datetime.now(), call_sid, status)
                )
                await db.commit()

    async def get_recent_calls_to_number(self, phone_number: str, within_period: timedelta) -> List[Dict]:
        """Get calls made to a number within a specific period."""
        cutoff_time = datetime.now() - within_period
        results = []
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT call_time, call_sid, status FROM number_call_history WHERE phone_number = ? AND call_time >= ? ORDER BY call_time DESC",
                    (phone_number, cutoff_time)
                ) as cursor:
                    async for row in cursor:
                        results.append({"call_time": row[0], "call_sid": row[1], "status": row[2]})
        return results

    # --- Method to update specific data fields ---
    async def update_call_data(self, call_sid: str, updates: Dict[str, Any]):
        """Updates specific data fields for an active call."""
        if call_sid in self.active_calls:
            state_machine = self.active_calls[call_sid]
            updated = False
            for key, value in updates.items():
                if hasattr(state_machine, key):
                    setattr(state_machine, key, value)
                    updated = True
                else:
                    logger.warning(f"Attempted to update non-existent field '{key}' for call {call_sid}")
            
            if updated:
                state_machine.last_update_time = time.time()
                # Optionally save immediately or rely on periodic/end-of-call save
                # await self.save_call_state(call_sid)
            return updated
        else:
            logger.warning(f"Cannot update data for inactive call {call_sid}")
            return False
            
    # --- Removed process_user_speech as LLM now drives flow ---
    # async def process_user_speech(...) 

    # --- NEW Methods for Dashboard --- 
    async def get_recent_calls(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Retrieves a list of recent calls with basic info."""
        calls = []
        sql = """
            SELECT call_sid, state, start_time, duration, from_number, to_number, is_outbound, script_name
            FROM calls 
            ORDER BY start_time DESC 
            LIMIT ? OFFSET ?
        """
        try:
            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row # Return rows as dict-like objects
                    async with db.execute(sql, (limit, offset)) as cursor:
                        async for row in cursor:
                            calls.append(dict(row))
            logger.debug(f"Retrieved {len(calls)} recent calls (limit={limit}, offset={offset}).")
        except Exception as e:
            logger.error(f"Error retrieving recent calls: {e}", exc_info=True)
        return calls

    async def get_call_details(self, call_sid: str) -> Optional[Dict]:
        """Retrieves all details for a specific call_sid."""
        details = None
        sql = "SELECT * FROM calls WHERE call_sid = ?"
        try:
            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(sql, (call_sid,)) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            details = dict(row)
                            # Potentially deserialize JSON fields here if needed by frontend
                            # Example:
                            # if 'history' in details and isinstance(details['history'], str):
                            #     try: details['history'] = json.loads(details['history'])
                            #     except json.JSONDecodeError: pass # Keep as string if invalid JSON
                            # if 'qualifiers_met' in details and isinstance(details['qualifiers_met'], str):
                            #     try: details['qualifiers_met'] = json.loads(details['qualifiers_met'])
                            #     except json.JSONDecodeError: pass
                            logger.debug(f"Retrieved details for call {call_sid}.")
                        else:
                            logger.warning(f"Call details not found in DB for call_sid: {call_sid}")
        except Exception as e:
            logger.error(f"Error retrieving call details for {call_sid}: {e}", exc_info=True)
        return details
    # --- End NEW Methods --- 

    def get_call_state_sync(self, call_sid: str) -> Optional[CallStateMachine]:
        """Synchronous version of get_call_state."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"No active call state found for SID {call_sid}")
                return None
            
            # Convert row tuple to dictionary based on column names
            column_names = [description[0] for description in cursor.description]
            data = dict(zip(column_names, row))
            return CallStateMachine.from_dict(data)
        except Exception as e:
            logger.error(f"Error getting call state for {call_sid}: {e}", exc_info=True)
            return None
        finally:
            conn.close()
    
    def create_call_state_sync(self, call_sid: str, initial_data: Dict[str, Any]) -> CallStateMachine:
        """Synchronous version of create_call_state."""
        # Create machine first
        state_machine = CallStateMachine(call_sid, ScriptState.START)
        
        # Set initial data
        if "from_number" in initial_data:
            state_machine.from_number = initial_data["from_number"]
        if "to_number" in initial_data:
            state_machine.to_number = initial_data["to_number"]
        if "is_outbound" in initial_data:
            state_machine.is_outbound = initial_data["is_outbound"]
        if "script_name" in initial_data:
            state_machine.script_name = initial_data["script_name"]
            
        # Insert into database - use the actual columns from the table schema
        state_dict = state_machine.to_dict()
        
        # Prepare SQL statement
        columns = ", ".join(state_dict.keys())
        placeholders = ", ".join(["?"] * len(state_dict))
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO calls ({columns}) VALUES ({placeholders})",
                tuple(state_dict.values())
            )
            conn.commit()
            logger.info(f"Created new call state for {call_sid}")
            return state_machine
        except Exception as e:
            logger.error(f"Error creating call state for {call_sid}: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def save_call_state_sync(self, call_sid: str) -> bool:
        """Synchronous version of save_call_state."""
        state_machine = self.get_call_state_sync(call_sid)
        if not state_machine:
            logger.error(f"Cannot save state for {call_sid}: No state machine found")
            return False
            
        # Update timestamp
        state_machine.last_update_time = time.time()
        state_machine.duration = state_machine.last_update_time - state_machine.start_time
        
        # Save to database
        state_dict = state_machine.to_dict()
        
        # Build SET clause for all columns
        set_clause = ", ".join([f"{col} = ?" for col in state_dict.keys()])
        values = list(state_dict.values())
        values.append(call_sid)  # Add call_sid for WHERE clause
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE calls SET {set_clause} WHERE call_sid = ?",
                tuple(values)
            )
            conn.commit()
            rows_updated = cursor.rowcount
            logger.debug(f"Saved call state for {call_sid}, rows updated: {rows_updated}")
            return rows_updated > 0
        except Exception as e:
            logger.error(f"Error saving call state for {call_sid}: {e}", exc_info=True)
            return False
        finally:
            conn.close()

    def end_call_sync(self, call_sid: str, final_state: ScriptState = ScriptState.CALL_ENDED_ABRUPTLY) -> bool:
        """Synchronous version of end_call. Marks a call as ended in the database."""
        # Get the current state
        state_machine = self.get_call_state_sync(call_sid)
        if not state_machine:
            logger.warning(f"Cannot end call {call_sid}: No active state found")
            return False
            
        # Update to final state
        state_machine.state = final_state
        state_machine.last_update_time = time.time()
        state_machine.duration = state_machine.last_update_time - state_machine.start_time
        
        # Save to database - we'll reuse the save_call_state_sync logic since we don't have a call_ended column
        # The state itself being CALL_ENDED_ABRUPTLY or COMPLETED marks it as ended
        return self.save_call_state_sync(call_sid)