"""
Telemarketer Server for Nova2

This module provides a FastAPI server that integrates with Twilio for:
- Inbound call handling with TwiML
- Outbound call management with queue and rate limiting
- Bidirectional WebSocket audio streaming
- Integration with Nova2's TTS and STT capabilities
- UK call regulations compliance
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from twilio.rest import Client
import base64 # For decoding Twilio media messages
import signal
import sys
import threading # Added import
import time # Added import
import sqlite3

# Import our components
from .stream_telemarketer import ScriptManager
from .call_state_manager import CallStateManager, ScriptState, ScriptAction
from .uk_call_regulations import get_regulator
# Import only what's needed from transcriptor_data
from .transcriptor_data import TranscriptorConditioning
from .tts_data import TTSConditioning
from .inference_engines import InferenceEngineXTTS, InferenceEngineElevenlabs, InferenceEngineGroq
from .security_manager import SecretsManager
from .API import NovaAPI
# Import context-related classes from context_data
from .context_data import ContextDatapoint, ContextSource_Voice 
# Import the manager class from its own file
from .context_manager import ContextManager
from .llm_data import LLMResponse, LLMConditioning
from .tool_data import LLMTool
# Import only the Message class, not SystemMessage or HumanMessage
from .llm_data import Message, Conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telemarketer_server")

# Add near the top with other environment variables
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# --- ADDED Debug Prints --- 
print("[DEBUG] Initializing components...")
print("[DEBUG] > Twilio Client...")
# Twilio credentials from environment
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_DEFAULT_FROM = os.environ.get("TWILIO_DEFAULT_FROM", "")
SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "http://localhost:8000")

# TTS Engine Selection
TTS_ENGINE_NAME = os.environ.get("TTS_ENGINE", "elevenlabs").lower()
DEFAULT_VOICE_NAME = os.environ.get("DEFAULT_VOICE_NAME", "agent_voice")

# Database path
DB_PATH = os.environ.get("DB_PATH", "telemarketer_calls.db")
TPS_PATH = os.environ.get("TPS_REGISTRY_PATH", None)

# Max concurrent calls
MAX_CONCURRENT_CALLS = int(os.environ.get("MAX_CONCURRENT_CALLS", "1"))

# --- NEW: STT Configuration ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small") # e.g., tiny, base, small, medium, large
# --- End STT Configuration ---

# --- Configuration Variables (Modify/Add) ---
TESTING_MODE = os.environ.get("TESTING_MODE", "twilio_stream").lower() # 'local' or 'twilio_stream'
LOCAL_TEST_CALL_SID = "local_test_001" # Fixed SID for local testing
# --- End Configuration Variables ---

# --- Constants ---
BASE_SYSTEM_PROMPT = """You are Isaac, a professional and friendly telemarketing assistant. 
Your goal is to follow the provided script instructions precisely for the current state, engage the user politely, and determine if they are a good fit for Proactiv's plastic card products (referral cards, loyalty cards, gift cards, MOT reminders, appointment cards). 
Gather necessary information and attempt to book a 10-15 minute virtual demo appointment.
Respond conversationally based *only* on the dialogue/instructions provided for the current state. 

TOOL USE:
You have access to the following tool:
- `end_call`: Use this tool ONLY when the conversation reaches a natural conclusion (e.g., appointment booked and farewell delivered, or user firmly declines and wishes to end conversation). 
  - Optional Parameter: `farewell_message` (string) - You can provide a custom final sentence to say before hanging up.

VALID STATE TRANSITIONS:
For the START state, you can only transition to: CONTINUE_INTRO, CALLBACK, or DECLINE.
Always pick a valid next state from the transitions allowed for the current state.

RESPONSE FORMAT:
After generating your conversational response, you MUST suggest the next logical script state based on the user's last message and the script flow by ending your entire response *exactly* with the tag: <next_state>DESIRED_NEXT_STATE_NAME</next_state>. Choose a state from the script structure provided. Do not add any text after the tag. 
If you use the `end_call` tool, you should still suggest the appropriate final state (e.g., FAREWELL or COMPLETED) using the <next_state> tag.
"""
# MAX_HISTORY_TURNS = 5 # REMOVE - Using built-in ctx_limit
# --- End Constants ---

# Initialize FastAPI app
app = FastAPI(title="Nova2 AI Telemarketer")

# Initialize Twilio client if credentials are available
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

print("[DEBUG] > Call State Manager...")
# Initialize call state manager
call_state_manager = CallStateManager(DB_PATH)

print("[DEBUG] > UK Regulator...")
# Initialize UK regulations
uk_regulator = get_regulator(DB_PATH, TPS_PATH)

print("[DEBUG] > Script Manager...")
# Initialize script manager (formerly streaming telemarketer)
telemarketer = ScriptManager()

print("[DEBUG] > Nova API...")
# Initialize Nova2 API
nova = NovaAPI()
print(f"[DEBUG] NovaAPI initialized. Object ID: {id(nova)}")

print("[DEBUG] Initializing call queue...")
# Call queue for outbound calls
call_queue = asyncio.PriorityQueue()

print("[DEBUG] Initializing active WS connections dict...")
# Active websocket connections
active_ws_connections: Dict[str, WebSocket] = {}

print("[DEBUG] Initializing call contexts dict...")
# Per-call Context Managers
call_contexts: Dict[str, ContextManager] = {}

print("[DEBUG] Initializing local test tasks dict...")
# Global handle for local test background tasks
local_test_tasks = {}
# --- End Global State ---

# Request models
# ... (Pydantic models) ...

# --- Global In-Memory State for Config --- 
# ... (current_tts_config definition) ...

# --- Load Tools (moved after Nova init, before configs) ---
print("[DEBUG] Loading Tools...")
loaded_tools: List[LLMTool] = []
try:
    logger.info("Loading LLM tools...")
    # Ensure nova is initialized before calling its methods
    loaded_tools = nova.load_tools(load_internal_tools=True) 
    logger.info(f"Loaded {len(loaded_tools)} tools: {[tool.name for tool in loaded_tools]}")
    # Verify end_call tool loaded
    if not any(tool.name == "end_call" for tool in loaded_tools):
         logger.warning("'end_call' tool was not loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tools: {e}", exc_info=True)

# --- Configure Voice Analysis (Transcriptor) --- 
print("[DEBUG] Configuring STT...")
try:
    logger.info(f"Configuring Whisper STT model size: {WHISPER_MODEL_SIZE}")
    transcriptor_conditioning = TranscriptorConditioning(
        model=WHISPER_MODEL_SIZE,  # Use configured model size
        device="cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu",
        language="en",
        microphone_index=10,   # TEMP: Hardcode to 10 to match unit test
        vad_threshold=0.40,    # Use the tuned threshold
        voice_boost=3.0      # Match unit test setting
    )
    
    print(f"[DEBUG] STT Conditioning Object Created: Mic Index={transcriptor_conditioning.microphone_index}") # Updated log
    print("[DEBUG][Configure STT] Calling nova.configure_transcriptor()...")
    nova.configure_transcriptor(conditioning=transcriptor_conditioning)
    print("[DEBUG][Configure STT] nova.configure_transcriptor() finished.")
    
    # Apply STT config immediately 
    print("[DEBUG][Configure STT] Calling nova.apply_config_transcriptor()...")
    nova.apply_config_transcriptor()
    print("[DEBUG][Configure STT] nova.apply_config_transcriptor() finished.")
    logger.info("STT Configured and Applied.")

    # Store conditioning for potential later use
    global stt_conditioning 
    stt_conditioning = transcriptor_conditioning

except Exception as e:
    logger.error(f"FATAL: Failed during STT configuration or application: {e}", exc_info=True)
    print(f"[ERROR] Failed STT setup: {e}")
# --- End STT Configuration ---

# --- Configure TTS --- 
print(f"[DEBUG] Configuring TTS... Using Nova object ID: {id(nova)}")
try:
    # ... (TTS config code) ...
    logger.info(f"Initial TTS config: Engine='{TTS_ENGINE_NAME}', Voice='{DEFAULT_VOICE_NAME}'")
    tts_engine = None
    tts_conditioning = None

    if TTS_ENGINE_NAME == "xtts":
        from Nova2.app.inference_engines.inference_tts.inference_xtts import InferenceEngineXTTS
        engine = InferenceEngineXTTS()  # Instantiate the engine class
        conditioning = TTSConditioning(
            model="tts_models/multilingual/multi-dataset/xtts_v2",
            voice="female",
            expressivness=0.75,
            stability=0.5,
            similarity_boost=0.75,
            use_speaker_boost=True
        )
    elif TTS_ENGINE_NAME == "elevenlabs":
        from Nova2.app.inference_engines.inference_tts.inference_elevenlabs import InferenceEngineElevenlabs
        # Get ElevenLabs key from security data
        key_manager = SecretsManager()
        elevenlabs_key = key_manager.get_secret(Secrets.ELEVENLABS_API)
        
        if not elevenlabs_key:
            print("[ERROR] ElevenLabs API key not found in secrets. Using XTTS fallback.")
            from Nova2.app.inference_engines.inference_tts.inference_xtts import InferenceEngineXTTS
            engine = InferenceEngineXTTS()  # Instantiate the engine class
            conditioning = TTSConditioning(
                model="tts_models/multilingual/multi-dataset/xtts_v2",
                voice="female",
                expressivness=0.75,
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=True
            )
        else:
            engine = InferenceEngineElevenlabs()  # Instantiate the engine class
            conditioning = TTSConditioning(
                model="eleven_flash_v2_5",  # Use the confirmed working model ID
                voice="JBFqnCBsd6RMkjVDRZzb",  # Use the working voice ID from example
                expressivness=0.75,
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=True
            )
    print(f"[DEBUG][Configure TTS] Before configure_tts: Engine type: {engine}, Conditioning type: {type(conditioning)}")

    nova.configure_tts(inference_engine=engine, conditioning=conditioning)
    print("[DEBUG][Configure TTS] AFTER nova.configure_tts call.")
    # --- Apply TTS config immediately (like notebook example) --- 
    try:
        print("[DEBUG][Configure TTS] Calling nova.apply_config_tts()...")
        nova.apply_config_tts()
        print("[DEBUG][Configure TTS] nova.apply_config_tts() finished.")
    except Exception as tts_apply_err:
         print(f"[DEBUG][Configure TTS] EXCEPTION during nova.apply_config_tts(): {tts_apply_err}")
         logger.error(f"FATAL: Failed to apply TTS config: {tts_apply_err}", exc_info=True)
         # Decide if we should exit or continue without TTS?
    # --- End Apply TTS config --- 
    
    # Don't apply config here
    # nova.apply_config_tts() # Removed this older comment
    logger.info(f"Initial TTS Engine '{TTS_ENGINE_NAME}' configured.")
except Exception as e:
    logger.error(f"FATAL: Failed to configure initial TTS engine: {e}", exc_info=True)

# --- Configure LLM --- 
print("[DEBUG] Configuring LLM...")
try:
    # ... (LLM config code) ...
    logger.info("Configuring LLM engine: Groq Llama 3 70b SpecDec")
    # 1. Get API Key
    groq_api_key = None
    try:
        # Fix: Use the Secrets enum instead of plain string
        from .security_data import Secrets
        groq_api_key = nova._security.get_secret(Secrets.GROQ_API)
    except Exception as key_err:
        print(f"[DEBUG] Error retrieving Groq API key: {key_err}")
        
    if not groq_api_key:
        print("[DEBUG] Groq API key not found in secrets. Using fallback configuration...")
        # Use a built-in local alternative or set a clear error flag
        llm_engine = None
        llm_conditioning = None
        logger.warning("No Groq API key found. LLM functionality will be limited.")
    else:    
        # 2. Instantiate Engine
        llm_engine = InferenceEngineGroq()
        
        # 3. Create Conditioning
        llm_conditioning = LLMConditioning(
            model="llama-3.3-70b-specdec", # Specify the desired model
            # Define other LLM parameters if needed (temperature, max_tokens etc.)
            # temperature=0.7, 
            # max_tokens=500, 
            add_default_sys_prompt=True,
            kwargs={"api_key": groq_api_key} # Pass API key securely
        )
    
    # 4. Configure Nova
    if llm_engine and llm_conditioning:
        nova.configure_llm(inference_engine=llm_engine, conditioning=llm_conditioning)
        
        # --- Apply LLM config immediately --- 
        try:
            print("[DEBUG][Configure LLM] Calling nova.apply_config_llm()...")
            nova.apply_config_llm()
            print("[DEBUG][Configure LLM] nova.apply_config_llm() finished.")
            logger.info("LLM Engine Groq (llama-3.3-70b-specdec) configured and ready.")
        except Exception as llm_apply_err:
             print(f"[DEBUG][Configure LLM] EXCEPTION during nova.apply_config_llm(): {llm_apply_err}")
             logger.error(f"FATAL: Failed to apply LLM config: {llm_apply_err}", exc_info=True)
    else:
        print("[DEBUG] Skipping LLM configuration due to missing API key or configuration.")
        # Ensure applications knows LLM isn't available
        
except Exception as e:
    logger.error(f"FATAL: Failed to configure LLM engine: {e}", exc_info=True)
    # Decide how to handle this - exit? For now, log and continue
    # raise SystemExit(f"Failed to initialize LLM: {e}") 

# --- Apply all Nova configurations --- 
# print("[DEBUG] Applying Nova configurations (STT, TTS, LLM)...") # Keep this commented out
# try:
#     nova.apply_config_all() # Apply all configs together
#     logger.info("Nova configurations applied.")
# except Exception as e:
#      logger.error(f"FATAL: Failed to apply Nova configurations: {e}", exc_info=True)

print("[DEBUG] Finished initial configuration, tool loading, and config application.")
# --- END ADDED Debug Prints --- 

# FastAPI routes
# ... (API routes) ...

# --- End NEW Control Endpoint ---

# --- Helper Functions ---

async def terminate_call(call_sid: str, farewell_message: Optional[str] = None):
    """
    Terminate a call with Twilio and clean up resources

    Args:
        call_sid: The Twilio call SID
        farewell_message: Optional farewell message to speak before ending
    """
    global audio_inputs # Declare usage of the global variable
    logger.info(f"[{call_sid}] Terminating call. Farewell: '{farewell_message}'")

    if TESTING_MODE == "local":
        # --- Local Mode Cleanup ---
        logger.info(f"[{call_sid}] Local mode: Stopping transcriptor and cleaning up tasks.")
        try:
            # TODO: Review nova.stop_transcriptor implementation for local mode
            nova.stop_transcriptor(context_id=call_sid) # Assumes stop_transcriptor exists and handles context_id
        except Exception as e:
            logger.error(f"[{call_sid}] Error stopping local transcriptor: {e}", exc_info=True)

        # Cancel any pending background tasks for this local call
        tasks_to_cancel = [task for name, task in local_test_tasks.items() if name.startswith(f"llm_turn_{call_sid}")]
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                logger.debug(f"[{call_sid}] Cancelled background task: {task.get_name()}")
        # Clean up task registry
        for name in list(local_test_tasks.keys()):
             if name.startswith(f"llm_turn_{call_sid}"):
                 del local_test_tasks[name]

        # Set final state in the global test_state_machine
        global test_state_machine
        if test_state_machine:
            final_state = ScriptState.CALL_ENDED_ABRUPTLY # Default
            if telemarketer.get_script_data(ScriptState.HANGUP):
                final_state = ScriptState.HANGUP
            test_state_machine.state = final_state
            logger.info(f"[{call_sid}] Call marked as ended with state {final_state.value}.")
        return # Don't proceed with Twilio/WebSocket logic

    # --- Twilio/WebSocket Mode Cleanup --- 
    # Stop STT for this call if running
    try:
        nova.stop_transcriptor(context_id=call_sid) # Assumes this works for Twilio mode too
        logger.info(f"[{call_sid}] VoiceAnalysis transcriptor stopped.")
    except Exception as e:
        logger.error(f"[{call_sid}] Error stopping transcriptor during termination: {e}", exc_info=True)

    # Clean up WebSocket connection if it exists
    ws = active_ws_connections.pop(call_sid, None)
    if ws:
        try:
            await ws.close()
            logger.info(f"[{call_sid}] Closed WebSocket connection.")
        except Exception as e:
            logger.error(f"[{call_sid}] Error closing WebSocket: {e}", exc_info=True)

    # Clean up audio input stream
    audio_input = audio_inputs.pop(call_sid, None)
    if audio_input:
        await audio_input.stop()
        logger.info(f"[{call_sid}] Stopped and removed custom audio input stream.")

    # Let end_call handle final state setting and saving
    final_state = ScriptState.CALL_ENDED_ABRUPTLY # Default
    if telemarketer.get_script_data(ScriptState.HANGUP):
        final_state = ScriptState.HANGUP
    await call_state_manager.end_call(call_sid, final_state=final_state)
    logger.info(f"[{call_sid}] Call marked as ended in state manager (Twilio mode) with state {final_state.value}.")

    # Send final command to Twilio to hang up
    try:
        if twilio_client:
            call = twilio_client.calls(call_sid).update(status='completed')
            logger.info(f"[{call_sid}] Requested Twilio to hang up call. New status: {call.status}")
        else:
            logger.warning(f"[{call_sid}] Twilio client not initialized, cannot send hangup command.")
    except Exception as e:
        logger.error(f"[{call_sid}] Failed to update Twilio call status to completed: {e}", exc_info=True)


async def process_llm_turn(call_sid: str, user_input: Optional[str], current_state: ScriptState, context_id: str):
    print(f"[DEBUG][process_llm_turn] Task started for call {call_sid}, state {current_state.value}") # Existing
    logger.debug(f"[{call_sid}] --- Processing LLM Turn --- State: {current_state.value}")
    script_text = telemarketer.get_script_text(current_state)
    script_action = telemarketer.get_script_action(current_state)
    logger.debug(f"[{call_sid}] Script Text (raw): {script_text[:100] if script_text else 'None'}...")
    logger.debug(f"[{call_sid}] Script Action: {script_action.value if script_action else 'None'}")

    # TODO: Replace get_context_history with actual context retrieval
    # history = nova.get_context_history(context_id)
    history = [] # Placeholder

    # Construct messages for LLM using the Message class
    llm_messages = [
        Message(author="system", content=BASE_SYSTEM_PROMPT),
        # TODO: Add chat history from context manager
        # messages.extend(history) # Assuming history is list of Message objects
    ]
    # Add current user input if provided
    if user_input:
        llm_messages.append(Message(author="user", content=user_input))
    else:
        # ... (logic to add pseudo-input for initial turn) ...
        if script_text and script_action == ScriptAction.TALK:
             llm_messages.append(Message(author="user", content=f"<system_instruction>You are now in the '{current_state.value}' state. Your goal is to deliver the following line naturally and then transition to the next logical state based on the user's potential response (or lack thereof if they don't speak):</system_instruction>\n\n{script_text}"))
        elif script_text and script_action == ScriptAction.LISTEN:
             llm_messages.append(Message(author="user", content=f"<system_instruction>You are now in the '{current_state.value}' state. Listen carefully to the user's response. Based on what they say, decide the next state.</system_instruction>"))
        
    # --- Create Conversation object --- 
    conversation_obj = Conversation(conversation=llm_messages)
    logger.debug(f"[{call_sid}] Total messages to LLM: {len(conversation_obj._conversation)}")
    # --- End Create Conversation object --- 

    try:
        # Check if LLM is properly initialized
        if not hasattr(nova, '_llm') or not nova._llm._inference_engine:
            print(f"[DEBUG][process_llm_turn] LLM not properly initialized")
            raise ValueError("LLM not properly initialized or configured")
            
        print(f"[DEBUG][process_llm_turn] Awaiting LLM response...") # ADDED
        logger.debug(f"[{call_sid}] Sending request to LLM...")
        
        # Fix: run_llm is a synchronous function, not async, so don't use await
        llm_response: LLMResponse = nova.run_llm(
            conversation=conversation_obj, # Pass Conversation object
            # context_id=context_id, # Pass context if nova.run_llm uses it (it doesn't)
            tools=loaded_tools
        )
        
        print(f"[DEBUG][process_llm_turn] Got LLM response.") # ADDED
        logger.debug(f"[{call_sid}] Raw LLM Response: {llm_response.message[:200] if llm_response.message else 'None'}...")

        dialogue_text = None # ADDED Initialize
        suggested_next_state_str = None
        if llm_response.tool_calls:
           logger.debug(f"[{call_sid}] LLM requested tool call: {llm_response.tool_calls[0].name}")
           if llm_response.tool_calls[0].name == "end_call":
                farewell = llm_response.tool_calls[0].arguments.get("farewell_message", "Okay, goodbye.")
                logger.info(f"[{call_sid}] LLM requested end_call. Terminating with message: '{farewell}'")
                await terminate_call(call_sid, farewell_message=farewell)
                return
           else:
                logger.warning(f"[{call_sid}] Received unhandled tool call: {llm_response.tool_calls[0].name}")
                dialogue_text = "I'm sorry, I can't do that right now."
                suggested_next_state_str = current_state.value
        else:
            print(f"[DEBUG][process_llm_turn] Parsing LLM response...") # ADDED
            response_content = llm_response.message
            dialogue_text, suggested_next_state_str = telemarketer.parse_llm_response(response_content)
            print(f"[DEBUG][process_llm_turn] Parsed. Dialogue: '{dialogue_text[:30]}...', Next State: {suggested_next_state_str}") # ADDED
            logger.debug(f"[{call_sid}] Parsed Dialogue: {dialogue_text[:100] if dialogue_text else 'None'}...")
            logger.debug(f"[{call_sid}] Parsed Next State: {suggested_next_state_str}")

    except Exception as e:
        print(f"[DEBUG][process_llm_turn] EXCEPTION during LLM/Parsing: {e}") # ADDED
        logger.error(f"[{call_sid}] Error getting LLM response or parsing it: {e}", exc_info=True)
        # Fallback dialogue for any error case
        dialogue_text = "I'm having trouble understanding right now. Could you please repeat that?"
        suggested_next_state_str = current_state.value # Suggest staying in current state on error

    # --- Get State Machine BEFORE validation --- 
    if TESTING_MODE == "local":
        global test_state_machine
        state_machine = test_state_machine
    else:
        state_machine = await call_state_manager.get_call_state(call_sid)
        
    if not state_machine:
         logger.error(f"[{call_sid}] Could not find state machine to validate/update state.")
         print(f"[DEBUG][process_llm_turn] ERROR: Could not find state machine.")
         return # Cannot proceed without state machine

    # --- Validate and Update State using CallStateMachine --- 
    print(f"[DEBUG][process_llm_turn] Validating and setting state: Current='{state_machine.state.value}', Suggested='{suggested_next_state_str}'")
    transition_successful = state_machine.validate_and_set_state(suggested_next_state_str)
    print(f"[DEBUG][process_llm_turn] Validation/Set Result: {transition_successful}. New State: {state_machine.state.value}")

    # Get the actual state *after* potential update
    next_state_enum = state_machine.state 

    # --- Update History in Manager (State already updated if valid) --- 
    print(f"[DEBUG][process_llm_turn] Adding history entry... Dialogue exists: {bool(dialogue_text)}")
    if dialogue_text: # Only add dialogue if it exists
        state_machine.add_history_entry("assistant", dialogue_text) 
        
    print(f"[DEBUG][process_llm_turn] Saving state machine (state='{state_machine.state.value}')...")
    if TESTING_MODE != "local":
        await call_state_manager.save_call_state(call_sid)
    print(f"[DEBUG][process_llm_turn] State machine saved.")
    # logger.info(f"[{call_sid}] Transitioned state: {current_state.value} -> {next_state_enum.value}") # Logged within validate_and_set_state
         
    # --- Speak the dialogue --- 
    if dialogue_text:
        print(f"[DEBUG][process_llm_turn] Speaking dialogue ('{dialogue_text[:30]}...')...")
        await handle_next_action(call_sid, dialogue_text, next_state_enum, context_id) 
        print(f"[DEBUG][process_llm_turn] Finished speaking dialogue.")
    # --- End Speak Dialogue --- 
             
    # --- Determine next action based on the potentially updated state --- 
    print(f"[DEBUG][process_llm_turn] Getting next action for state {next_state_enum.value}...")
    next_action_enum = telemarketer.get_script_action(next_state_enum) # Use ScriptManager to get action
    if next_action_enum == ScriptAction.TALK:
        # Dialogue was already spoken above
        logger.debug(f"[{call_sid}] Next action is TALK. (Dialogue already spoken)")
        print(f"[DEBUG][process_llm_turn] Action is TALK, dialogue already spoken.") # ADDED
        # Remove the call here as it's done earlier
        # await handle_next_action(call_sid, dialogue_text, next_state_enum, context_id)
    elif next_action_enum == ScriptAction.LISTEN:
        logger.debug(f"[{call_sid}] Next action is LISTEN. Waiting for input.")
        print(f"[DEBUG][process_llm_turn] Action is LISTEN. No action needed.") # Existing
    elif next_action_enum == ScriptAction.HANGUP:
        logger.debug(f"[{call_sid}] Next action is HANGUP. Terminating.")
        farewell_msg = telemarketer.get_script_text(next_state_enum) or "Okay, goodbye."
        print(f"[DEBUG][process_llm_turn] Calling terminate_call...") # ADDED
        await terminate_call(call_sid, farewell_msg)
        print(f"[DEBUG][process_llm_turn] terminate_call finished.") # ADDED

    logger.debug(f"[{call_sid}] --- Finished LLM Turn --- State: {next_state_enum.value}")
    print(f"[DEBUG][process_llm_turn] Task finished for state {current_state.value} -> {next_state_enum.value}.") # ADDED


# Add a global in-memory state machine at the top after initializing call_state_manager
call_state_manager = CallStateManager(DB_PATH)
print(f"[DEBUG] > Call State Manager...")

# Add this line
test_state_machine = None  # Global variable for test mode

# Then update the stt_callback function
async def stt_callback(text: str, context_id: str):
    """Callback function triggered by VoiceAnalysis when transcription is ready."""
    logger.info(f"[{context_id}] STT Callback Received: '{text}'")
    call_sid = context_id

    # Use global test_state_machine for local test mode
    if TESTING_MODE == "local":
        global test_state_machine
        state_machine = test_state_machine
    else:
        state_machine = await call_state_manager.get_call_state(call_sid)
        
    if not state_machine:
        logger.error(f"[{call_sid}] Received STT callback for unknown or ended call.")
        return
        
    current_state = state_machine.state
    current_action = telemarketer.get_script_action(current_state)

    if current_action == ScriptAction.LISTEN:
        logger.info(f"[{call_sid}] Processing user input '{text}' received in state '{current_state.value}'")
        # --- Add user input to history --- 
        state_machine.add_history_entry("user", text)
        if TESTING_MODE != "local":
            await call_state_manager.save_call_state(call_sid) # Save state with new history
        # --- End Add user input --- 

        task_name = f"llm_turn_{call_sid}_{uuid.uuid4()}"
        task = asyncio.create_task(process_llm_turn(call_sid, user_input=text, current_state=current_state, context_id=context_id), name=task_name)
        if TESTING_MODE == 'local':
            local_test_tasks[task_name] = task
    else:
        logger.warning(f"[{call_sid}] STT callback received text '{text}' but current state is '{current_state.value}' ({current_action.value if current_action else 'None'}), ignoring input.")

# --- End Helper Functions ---

# --- ADD handle_next_action definition here ---
async def handle_next_action(call_sid: str, text_to_speak: str, current_state: ScriptState, context_id: str):
    """Handle the next action for a call: Speak message via TTS."""
    print(f"[DEBUG][handle_next_action] Entered for state {current_state.value}")
    output_target = "local_speaker" if TESTING_MODE == "local" else "twilio_stream"
    logger.debug(f"[{call_sid}] Speaking in state '{current_state.value}'. Output target: {output_target}")

    if not text_to_speak:
        logger.warning(f"[{call_sid}] handle_next_action called with no text to speak for state {current_state.value}.")
        return
        
    print(f"[DEBUG][handle_next_action] Starting TTS process for text: '{text_to_speak[:30]}...'")
    
    # --- Use the provided text_to_speak for TTS --- 
    logger.info(f"[{call_sid}] Generating TTS for state '{current_state.value}' snippet: '{text_to_speak[:50]}...'")
    
    # Create variable to track audio duration for wait timing
    audio_duration = 0
    
    # Use nova instance (which has configured TTS engine)
    try:
        print(f"[DEBUG][handle_next_action] Calling nova.run_tts(stream=True)...")
        tts_audio_stream = nova.run_tts(text=text_to_speak, stream=True) # Use stream=True for faster response
        print(f"[DEBUG][handle_next_action] Streaming TTS completed successfully")
    except Exception as stream_err:
        print(f"[DEBUG][handle_next_action] ERROR in streaming TTS: {stream_err}")
        logger.error(f"[{call_sid}] Failed to generate streaming TTS: {stream_err}", exc_info=True)
        return

    if output_target == "local_speaker":
        # --- Local Testing: Play audio to speakers --- 
        print(f"[DEBUG][handle_next_action] LOCAL MODE: Calling nova.run_tts(stream=False)...")
        audio_data_obj = nova.run_tts(text=text_to_speak, stream=False) # Get non-streaming for playback
        print(f"[DEBUG][handle_next_action] LOCAL MODE: nova.run_tts finished. AudioData received: {bool(audio_data_obj)}")
        if audio_data_obj:
            print(f"[DEBUG][handle_next_action] AudioData object type: {type(audio_data_obj)}")
            if hasattr(audio_data_obj, '_audio_data'):
                print(f"[DEBUG][handle_next_action] _audio_data attribute exists and is type: {type(audio_data_obj._audio_data)}")
                
                # Try to get audio data details for debugging
                try:
                    if hasattr(audio_data_obj._audio_data, 'duration_seconds'):
                        print(f"[DEBUG][handle_next_action] Audio duration: {audio_data_obj._audio_data.duration_seconds} seconds")
                        audio_duration = audio_data_obj._audio_data.duration_seconds
                    if hasattr(audio_data_obj._audio_data, 'frame_rate'):
                        print(f"[DEBUG][handle_next_action] Audio frame rate: {audio_data_obj._audio_data.frame_rate}")
                    if hasattr(audio_data_obj._audio_data, 'channels'):
                        print(f"[DEBUG][handle_next_action] Audio channels: {audio_data_obj._audio_data.channels}")
                    if hasattr(audio_data_obj._audio_data, 'sample_width'):
                        print(f"[DEBUG][handle_next_action] Audio sample width: {audio_data_obj._audio_data.sample_width}")
                    print(f"[DEBUG][handle_next_action] Audio data size: {len(audio_data_obj._audio_data.raw_data) if hasattr(audio_data_obj._audio_data, 'raw_data') else 'unknown'} bytes")
                except Exception as audio_attr_err:
                    print(f"[DEBUG][handle_next_action] Error accessing audio attributes: {audio_attr_err}")
                    
            else:
                print(f"[DEBUG][handle_next_action] WARNING: _audio_data attribute is missing!")
                
            print(f"[DEBUG][handle_next_action] Calling nova.play_audio...")
            try:
                nova.play_audio(audio_data_obj)
                print(f"[DEBUG][handle_next_action] nova.play_audio finished.")
                
                # Add a short delay to ensure playback completes before resuming STT
                print(f"[DEBUG][handle_next_action] Waiting for playback to complete...")
                # Use the actual audio duration if available, otherwise use default
                wait_duration = audio_duration if audio_duration > 0 else 5
                # Add some buffer time to ensure audio playback is fully complete
                await asyncio.sleep(wait_duration + 1.0)
                print(f"[DEBUG][handle_next_action] Finished waiting after audio playback ({wait_duration + 1.0}s)")
                
            except Exception as audio_play_err:
                print(f"[DEBUG][handle_next_action] EXCEPTION during nova.play_audio call: {audio_play_err}")
                print(f"[DEBUG][handle_next_action] Error type: {type(audio_play_err)}")
                
                # Fallback to file saving if playback fails
                try:
                    # Create tmp_output directory if it doesn't exist
                    os.makedirs(os.path.join(os.getcwd(), "tmp_output"), exist_ok=True)
                    # Save the file with a cleaned filename
                    safe_filename = ''.join(c if c.isalnum() else '_' for c in text_to_speak[:20])
                    output_path = os.path.join(os.getcwd(), "tmp_output", f"tts_output_{call_sid}_{safe_filename}.mp3")
                    print(f"[DEBUG][handle_next_action] Fallback: Saving audio to file: {output_path}")
                    audio_data_obj._audio_data.export(output_path, format="mp3")
                    print(f"[DEBUG][handle_next_action] Audio saved to {output_path}")
                except Exception as save_err:
                    print(f"[DEBUG][handle_next_action] Error saving audio file: {save_err}")
        else:
            print(f"[DEBUG][handle_next_action] WARNING: No audio data object received from TTS!")
    else: # Default: Twilio Stream Mode (output_target == "twilio_stream")
        ws = active_ws_connections.get(call_sid)
        if not ws:
            logger.error(f"[{call_sid}] WebSocket missing for Twilio stream output in handle_next_action.")
            return
        # --- Twilio Stream: Send audio over WebSocket --- 
        logger.debug(f"[{call_sid}] Sending TTS audio stream to Twilio WebSocket.")
        media_message = {
            "event": "media",
            "streamSid": call_sid, 
            "media": {"payload": ""}
        }
        # Use the streaming iterator directly
        async for audio_chunk in tts_audio_stream:
            if audio_chunk:
                try:
                    # TODO: Add audio conversion (e.g., to mulaw) if required by Twilio
                    converted_chunk = audio_chunk # Assuming raw stream for now
                    
                    payload = base64.b64encode(converted_chunk).decode('utf-8')
                    media_message["media"]["payload"] = payload
                    await ws.send_text(json.dumps(media_message))
                except WebSocketDisconnect:
                    logger.warning(f"[{call_sid}] WebSocket disconnected while sending TTS audio.")
                    break
                except Exception as e:
                    logger.error(f"[{call_sid}] Error sending TTS chunk to WebSocket: {e}", exc_info=True)
                    break 
        logger.debug(f"[{call_sid}] Finished sending TTS audio stream.")
        # --- End Twilio Stream Sending --- 
    
    # Resume STT after speaking with a short delay to avoid self-feedback
    await asyncio.sleep(0.2)  # Small delay before resuming STT

# --- END handle_next_action definition ---

# --- Updated Async STT Processor --- 
async def process_stt_async(generator, context_id):
    """Runs as an async task, processing STT generator output and calling callback.
       Uses asyncio.to_thread to avoid blocking the main loop if generator waits.
    """
    try:
        logger.info(f"[{context_id}] Async STT processing task started.")
        print(f"[DEBUG][process_stt_async][{context_id}] >>> Entering async for loop...")
        
        # Iterate directly over the generator
        # If the generator method itself blocks internally (e.g., queue.get), 
        # this loop might still block. Consider wrapping the loop body or the 
        # generator retrieval if blocking occurs.
        async for datapoint in generator:
            print(f"[DEBUG][process_stt_async][{context_id}] >>> Generator yielded item: {type(datapoint)}") 
            if isinstance(datapoint, ContextDatapoint) and hasattr(datapoint, 'source') and isinstance(datapoint.source, ContextSource_Voice):
                text = datapoint.content
                if text:
                    # Directly await the callback since we are in an async function
                    try:
                        await stt_callback(text, context_id)
                    except Exception as callback_err:
                         logger.error(f"[{context_id}] Error during stt_callback: {callback_err}", exc_info=True)
            # Yield control briefly to the event loop
            await asyncio.sleep(0.01) 
            
    except Exception as e:
        logger.error(f"[{context_id}] Error in async STT processing task: {e}", exc_info=True)
    finally:
        logger.info(f"[{context_id}] Async STT processing task finished.")
# --- END Updated Async STT Processor --- 

# --- Main Async Function for Local Test --- 
def run_local_test():
    """
    Start a test call locally that uses the microphone and speakers
    """
    print("[DEBUG][run_local_test] Entered run_local_test.")
    
    # Use a completely in-memory approach
    global test_state_machine
    test_state_machine = CallStateMachine(LOCAL_TEST_CALL_SID, ScriptState.START)
    test_state_machine.from_number = "local_mic"
    test_state_machine.to_number = "local_speaker"
    test_state_machine.is_outbound = False
    test_state_machine.script_name = "local_test_script"
    
    print(f"[DEBUG][run_local_test] Created in-memory state machine with state {test_state_machine.state.value}")
    
    print("[DEBUG][run_local_test] Starting STT directly...")
    try:
        # Initialize VoiceAnalysis directly to avoid any NovaAPI issues
        if not hasattr(nova, '_stt') or not nova._stt:
            print("[DEBUG][run_local_test] ERROR: nova._stt not initialized!")
            return
            
        # Get the transcriptor object
        transcriptor = nova._stt
        
        # Call start_sync method directly 
        print("[DEBUG][run_local_test] Calling transcriptor.start_sync() directly...")
        stt_generator = transcriptor.start_sync()
        
        if stt_generator:
            print("[DEBUG][run_local_test] Got sync generator, now processing...")
            
            # Simple blocking loop to consume from generator directly
            print("[DEBUG][run_local_test] Starting direct processing loop...")
            try:
                # Iteration through a limited number for testing
                for i, datapoint in enumerate(stt_generator):
                    if i >= 100:  # Safety limit
                        break
                        
                    print(f"[DEBUG][run_local_test] Received datapoint: {datapoint.content if hasattr(datapoint, 'content') else 'No content'}")
                    time.sleep(0.1)  # Small sleep to avoid CPU spinning
            except KeyboardInterrupt:
                print("[DEBUG][run_local_test] KeyboardInterrupt detected, exiting loop")
            except Exception as e:
                print(f"[DEBUG][run_local_test] ERROR processing STT output: {e}")
        else:
            print("[DEBUG][run_local_test] start_sync() returned None!")
            
    except Exception as e:
        print(f"[DEBUG][run_local_test] ERROR starting STT: {e}")
    finally:
        # Always clean up
        if hasattr(nova, '_stt') and hasattr(nova._stt, 'close'):
            print("[DEBUG][run_local_test] Calling transcriptor.close() to clean up...")
            try:
                nova._stt.close()
                print("[DEBUG][run_local_test] Transcriptor closed successfully.")
            except Exception as close_err:
                print(f"[DEBUG][run_local_test] ERROR closing transcriptor: {close_err}")
    
    print("[DEBUG][run_local_test] Exiting...")
    return

# --- Local Test Mode Shutdown --- 
def shutdown_local_test():
    """Minimal synchronous cleanup function for local test mode."""
    print("[DEBUG][shutdown_local_test] Cleaning up...")
    
    # Clean up STT if it exists
    if hasattr(nova, '_stt') and nova._stt:
        try:
            nova._stt.close()
            print("[DEBUG][shutdown_local_test] Closed transcriptor")
        except Exception as e:
            print(f"[DEBUG][shutdown_local_test] Error closing transcriptor: {e}")
            
    print("[DEBUG][shutdown_local_test] Done")

# --- Main Execution / Server Start --- 
if __name__ == "__main__":
    print("[DEBUG] Entered __main__ block.") 
    print(f"[DEBUG] TESTING_MODE is '{TESTING_MODE}'. Attempting to start...")

    if TESTING_MODE == "local":
        try:
            run_local_test()  # Direct call, no asyncio.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            logger.info("Running local shutdown procedure...")
            shutdown_local_test()  # Direct call, no asyncio.run()
            logger.info("Local test shutdown complete.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

    logger.info("Telemarketer Server stopped.") 