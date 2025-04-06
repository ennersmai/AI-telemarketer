#!/usr/bin/env python
"""
Nova Server Implementation

A complete implementation of the Nova server that integrates all components
including speech-to-text, text-to-speech, LLM, and conversation management.
"""

import logging
import os
import sys
import threading
import time
import queue
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Nova2 imports
from Nova2.app.transcriptor import VoiceAnalysis, TranscriptorConditioning
from Nova2.app.tts_manager import TTSManager
from Nova2.app.tts_data import TTSConditioning
from Nova2.app.audio_manager import AudioPlayer
from Nova2.app.context_manager import ContextManager, Context
from Nova2.app.event_system import EventSystem, EventListener
from Nova2.app.stream_telemarketer import ScriptManager
from Nova2.app.API import NovaAPI
from Nova2.app.llm_data import Conversation, Message, LLMResponse, MemoryConfig
from Nova2.app.context_data import ContextDatapoint, ContextSource_Assistant, ContextSource_User
from Nova2.app.security_data import Secrets
from Nova2.app.tool_data import LLMTool
from Nova2.app.call_state_manager import ScriptState, ScriptAction, CallStateMachine
from Nova2.app.script_parser import load_script_section

# Import inference engines
from Nova2.app.inference_engines import (
    InferenceEngineLlamaCPP,
    InferenceEngineGroq,
    InferenceEngineElevenlabs
)

# Import conditioning
from Nova2.app.llm_data import LLMConditioning

class EnhancedScriptManager:
    """
    Enhanced script manager that loads dialogues from telemarketer_script.md file
    and manages state transitions properly based on the full script flow.
    """
    
    def __init__(self, script_dir: Optional[str] = None):
        """
        Initialize the Enhanced Script Manager with script loading from MD files.
        
        Args:
            script_dir: Directory containing script files. If None, uses default path.
        """
        self.script_dir = script_dir or os.path.join(Path(__file__).parent.parent, "data", "scripts")
        self.current_script = "telemarketer_script.md"
        self.script_path = os.path.join(self.script_dir, self.current_script)
        self.script_data = {}
        self.available_states = []
        
        # Load scripts
        self._load_all_state_definitions()
        logger.info(f"EnhancedScriptManager initialized with {len(self.available_states)} available states")
    
    def _load_all_state_definitions(self):
        """Load all state definitions by checking each ScriptState enum value."""
        self.available_states = []
        
        # Iterate through all ScriptState enum values
        for state in ScriptState:
            state_name = state.value
            dialogue = load_script_section(state_name, self.script_path)
            
            if dialogue:
                # Store the dialogue and add state to available states
                self.script_data[state_name] = {
                    "dialogue": dialogue,
                    "state": state
                }
                self.available_states.append(state_name)
                logger.debug(f"Loaded dialogue for state: {state_name}")
        
        logger.info(f"Loaded {len(self.available_states)} states with dialogues")
    
    def load_script(self, script_name: str) -> bool:
        """
        Load a specific script by name.
        
        Args:
            script_name: Name of the script file (with or without .md extension)
            
        Returns:
            True if script was loaded successfully, False otherwise
        """
        if not script_name.endswith(".md"):
            script_name = f"{script_name}.md"
        
        script_path = os.path.join(self.script_dir, script_name)
        if not os.path.exists(script_path):
            logger.error(f"Script file not found: {script_path}")
            return False
        
        self.current_script = script_name
        self.script_path = script_path
        self._load_all_state_definitions()
        return True
    
    def get_prompt_for_state(self, state: str) -> Optional[str]:
        """
        Get the prompt/dialogue for a specific state.
        
        Args:
            state: The state name
            
        Returns:
            The dialogue string for the state, or None if not found
        """
        state_data = self.script_data.get(state)
        if state_data:
            return state_data["dialogue"]
        return None
    
    def get_available_states(self) -> List[str]:
        """
        Get a list of all available states in the script.
        
        Returns:
            List of state names
        """
        return self.available_states
    
    def extract_next_state(self, llm_response: str) -> Optional[str]:
        """
        Extract the next state from an LLM response.
        Looks for explicit state mention or a tag.
        
        Args:
            llm_response: The LLM response text
            
        Returns:
            The extracted state name, or None if not found
        """
        # Try to find an explicit next_state tag
        match = re.search(r"<next_state>(.*?)</next_state>\s*$", llm_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().upper()
        
        # Try to find "next state: STATE_NAME" pattern
        match = re.search(r"next state:\s*(\w+)", llm_response.lower())
        if match:
            return match.group(1).upper()
        
        return None

class NovaServer:
    """
    Complete Nova Server implementation that integrates speech-to-text, text-to-speech,
    LLM processing, memory, tools, and conversation management.
    """
    
    def __init__(self, use_llm=True, use_tts=True, use_memory=True, use_tools=True):
        """
        Initialize the Nova Server with all required components.
        
        Args:
            use_llm: Whether to configure and use LLM
            use_tts: Whether to configure and use TTS
            use_memory: Whether to enable memory retrieval
            use_tools: Whether to load and use tools
        """
        logger.info("Initializing Nova Server...")
        
        # Options
        self.use_llm = use_llm
        self.use_tts = use_tts
        self.use_memory = use_memory
        self.use_tools = use_tools
        
        # Initialize Nova API
        self.nova = NovaAPI()
        
        # Initialize context manager
        self.context_manager = ContextManager()
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Set up components
        self._setup_audio_components()
        
        if self.use_llm:
            self._setup_llm()
        
        if self.use_tts:
            self._setup_tts()
        
        if self.use_tools:
            self._load_tools()
        
        if self.use_memory:
            self._setup_memory()
        
        # Initialize enhanced script manager
        self.script_manager = EnhancedScriptManager()
        
        # Initialize state machine for telemarketer flow
        self.call_state_machine = CallStateMachine("local_session")
        
        # State tracking
        self.current_state = ScriptState.GREETING.value
        self.conversation_history = []
        self.is_running = False
        self.processing_input = False
        
        # Threading resources
        self.stt_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.llm_thread = None
        
        # Register event listeners
        self._register_event_listeners()
        
        logger.info("Nova Server initialization complete")
    
    def _setup_audio_components(self):
        """Set up audio input/output components (STT, AudioPlayer)."""
        # Get microphone index from environment or use default
        mic_index = int(os.environ.get("MICROPHONE_INDEX", 0))
        
        # Configure STT
        logger.info(f"Configuring speech-to-text with microphone index {mic_index}")
        stt_conditioning = TranscriptorConditioning(
            model="small",
            device="cuda" if not os.environ.get("FORCE_CPU") else "cpu",
            language="en",
            microphone_index=mic_index,
            vad_threshold=0.40,
            voice_boost=3.0
        )
        
        # Initialize STT in the Nova API
        self.nova.configure_transcriptor(stt_conditioning)
        self.nova.apply_config_transcriptor()
    
    def _setup_llm(self):
        """Set up the LLM component."""
        logger.info("Setting up LLM...")
        
        try:
            # Determine the inference engine to use
            engine_type = os.environ.get("LLM_ENGINE", "llamacpp").lower()
            
            if engine_type == "groq":
                # Configure Groq API
                logger.info("Using Groq for LLM inference")
                inference_engine = InferenceEngineGroq()
                conditioning = LLMConditioning(
                    model=os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
                )
                # Set API key if provided
                api_key = os.environ.get("GROQ_API_KEY")
                if api_key:
                    self.nova.edit_secret(Secrets.GROQ_API, api_key)
            else:
                # Configure LlamaCPP (default)
                logger.info("Using LlamaCPP for LLM inference")
                inference_engine = InferenceEngineLlamaCPP()
                conditioning = LLMConditioning(
                    model=os.environ.get("LLM_MODEL", "bartowski/Qwen2.5-7B-Instruct-1M-GGUF"),
                    file=os.environ.get("LLM_FILE", "*Q8_0.gguf")
                )
                
                # Set HF token if provided
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    self.nova.huggingface_login(overwrite=True, token=hf_token)
            
            # Configure LLM
            self.nova.configure_llm(inference_engine=inference_engine, conditioning=conditioning)
            self.nova.apply_config_llm()
            logger.info("LLM setup complete")
            
        except Exception as e:
            logger.error(f"Failed to set up LLM: {e}", exc_info=True)
            self.use_llm = False
    
    def _setup_tts(self):
        """Set up the TTS component."""
        logger.info("Setting up TTS...")
        
        try:
            # Determine the inference engine to use
            engine_type = os.environ.get("TTS_ENGINE", "elevenlabs").lower()
            
            if engine_type == "elevenlabs":
                # Configure ElevenLabs
                logger.info("Using ElevenLabs for TTS inference")
                inference_engine = InferenceEngineElevenlabs()
                conditioning = TTSConditioning(
                    model=os.environ.get("TTS_MODEL", "eleven_multilingual_v2"),
                    voice=os.environ.get("TTS_VOICE", "21m00Tcm4TlvDq8ikWAM"),  # Default to "Rachel"
                    expressivness=float(os.environ.get("TTS_EXPRESSIVENESS", "0.5")),
                    stability=float(os.environ.get("TTS_STABILITY", "0.5")),
                    similarity_boost=0.75,
                    use_speaker_boost=True
                )
                
                # Set API key if provided
                api_key = os.environ.get("ELEVENLABS_API_KEY")
                if api_key:
                    self.nova.edit_secret(Secrets.ELEVENLABS_API, api_key)
            else:
                # Use ElevenLabs as fallback
                logger.warning(f"Unknown TTS engine: {engine_type}. Using ElevenLabs instead.")
                inference_engine = InferenceEngineElevenlabs()
                conditioning = TTSConditioning(
                    model=os.environ.get("TTS_MODEL", "eleven_multilingual_v2"),
                    voice=os.environ.get("TTS_VOICE", "21m00Tcm4TlvDq8ikWAM"),  # Default to "Rachel"
                    expressivness=float(os.environ.get("TTS_EXPRESSIVENESS", "0.5")),
                    stability=float(os.environ.get("TTS_STABILITY", "0.5")),
                    similarity_boost=0.75,
                    use_speaker_boost=True
                )
                
                # Set API key if provided
                api_key = os.environ.get("ELEVENLABS_API_KEY")
                if api_key:
                    self.nova.edit_secret(Secrets.ELEVENLABS_API, api_key)
            
            # Configure TTS
            self.nova.configure_tts(inference_engine=inference_engine, conditioning=conditioning)
            self.nova.apply_config_tts()
            logger.info("TTS setup complete")
            
        except Exception as e:
            logger.error(f"Failed to set up TTS: {e}", exc_info=True)
            self.use_tts = False
    
    def _load_tools(self):
        """Load tools for LLM to use."""
        logger.info("Loading tools...")
        
        try:
            # Load all available tools
            self.tools = self.nova.load_tools(
                load_internal_tools=True,
                # Optional: include/exclude specific tools
                # include=["tool1", "tool2"],
                # exclude=["tool3", "tool4"]
            )
            logger.info(f"Loaded {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to load tools: {e}", exc_info=True)
            self.tools = []
            self.use_tools = False
    
    def _setup_memory(self):
        """Configure memory for LLM."""
        logger.info("Setting up memory...")
        
        try:
            # Configure memory settings
            self.memory_config = MemoryConfig(
                retrieve_memories=True,
                num_results=int(os.environ.get("MEMORY_NUM_RESULTS", "5")),
                search_area=int(os.environ.get("MEMORY_SEARCH_AREA", "2")),
                cosine_threshold=float(os.environ.get("MEMORY_THRESHOLD", "0.7"))
            )
            logger.info("Memory setup complete")
            
        except Exception as e:
            logger.error(f"Failed to set up memory: {e}", exc_info=True)
            self.memory_config = None
            self.use_memory = False
    
    def _register_event_listeners(self):
        """Register listeners for various events in the system."""
        # Register for transcription events
        self.event_system.register(
            "transcription_received", 
            EventListener(self._process_transcription)
        )
        
        # Register for LLM response events
        self.event_system.register(
            "llm_response_received",
            EventListener(self._handle_llm_response)
        )
        
        # Register for TTS completion events
        self.event_system.register(
            "tts_completed",
            EventListener(self._handle_tts_completed)
        )
        
        # Register for state change events
        self.event_system.register(
            "state_changed",
            EventListener(self._handle_state_changed)
        )
    
    def start_session(self):
        """Start a new conversation session."""
        try:
            logger.info("Starting Nova session")
            self.is_running = True
            
            # Load initial script
            self.script_manager.load_script("default")
            self.current_state = "greeting"
            
            # Initialize context with session info
            session_id = str(time.time())
            self.context_manager.set_context("session_id", session_id)
            self.context_manager.set_context("start_time", time.time())
            
            # Start transcriptor and bind to context
            logger.info("Starting speech-to-text transcriptor")
            try:
                # Give time for other components to initialize fully
                time.sleep(1.0)
                # Start in sync mode instead of async to avoid potential deadlocks
                context_generator = self.nova.start_transcriptor_sync(context_id=session_id)
                if context_generator:
                    self.nova.bind_context_source(context_generator)
                    logger.info("Transcriptor started successfully and bound to context")
                else:
                    logger.warning("Failed to start transcriptor - no generator returned")
            except Exception as stt_err:
                logger.error(f"Error starting transcriptor: {stt_err}", exc_info=True)
                # Continue even if transcriptor fails - we'll rely on other input methods
            
            # Process initial state (greeting)
            initial_prompt = self.script_manager.get_prompt_for_state(self.current_state)
            if initial_prompt:
                self._process_llm_turn(initial_prompt, is_system=True)
            
            # Main server loop
            while self.is_running:
                try:
                    # Process any pending STT results
                    self._process_queued_items()
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"Error in main server loop: {e}")
                    
            logger.info("Nova session complete")
        finally:
            self._cleanup()
    
    def _process_queued_items(self):
        """Process any items in the STT or LLM queues."""
        # Process STT queue
        try:
            while not self.stt_queue.empty():
                transcription = self.stt_queue.get_nowait()
                self._process_transcription(transcription)
                self.stt_queue.task_done()
        except queue.Empty:
            pass
        
        # Process LLM queue if we're not already processing
        if not self.processing_input and not self.llm_queue.empty():
            try:
                llm_input = self.llm_queue.get_nowait()
                self._start_llm_processing(llm_input)
                self.llm_queue.task_done()
            except queue.Empty:
                pass
    
    def _handle_transcription(self, transcription: str):
        """Callback for the VoiceAnalysis transcribe_callback."""
        logger.debug(f"Transcription received: {transcription}")
        self.stt_queue.put(transcription)
    
    def _process_transcription(self, transcription: str):
        """Process a transcription received from STT."""
        if not transcription or len(transcription.strip()) == 0:
            return
        
        logger.info(f"Processing transcription: {transcription}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": transcription})
        
        # Add to context manually if not already handled by context generator
        datapoint = ContextDatapoint(
            source=ContextSource_User(),
            content=transcription
        )
        self.nova.add_datapoint_to_context(datapoint)
        
        # Queue for LLM processing
        self.llm_queue.put(transcription)
        
        # Fire event
        self.event_system.fire("transcription_received", transcription)
    
    def _start_llm_processing(self, user_input: str):
        """Start processing user input with the LLM in a separate thread."""
        if self.processing_input:
            logger.warning("Already processing input, skipping")
            return
        
        self.processing_input = True
        
        # Use a thread to avoid blocking the main thread
        self.llm_thread = threading.Thread(
            target=self._process_llm_turn,
            args=(user_input,),
            daemon=True
        )
        self.llm_thread.start()
    
    def _process_llm_turn(self, user_input: str, is_system: bool = False):
        """
        Process a turn with the LLM, generating a response and next state.
        
        Args:
            user_input: The user's transcribed speech or system prompt
            is_system: Whether this is a system-initiated turn
        """
        try:
            logger.info(f"Processing {'system' if is_system else 'user'} input: {user_input}")
            
            # Get current script and state info
            current_script = self.script_manager.current_script
            available_states = self.script_manager.get_available_states()
            
            # Get current state from the state machine
            current_state = self.call_state_machine.state.value
            
            # Prepare context for LLM
            context = {
                "current_state": current_state,
                "available_states": available_states,
                "conversation_history": self.conversation_history,
                "user_input": user_input,
                "is_system_turn": is_system
            }
            
            # Update global context
            for key, value in context.items():
                self.context_manager.set_context(key, value)
            
            if self.use_llm:
                # Create conversation from history or get from context
                context_obj = self.nova.get_context()
                
                # Either use context conversation or create a new one
                if self.use_memory:
                    conversation = context_obj.to_conversation()
                else:
                    conversation = Conversation()
                    
                    # Add telemarketer-specific system prompt
                    system_prompt = f"""You are Isaac, a professional and friendly telemarketing assistant following a script.
Current state: {current_state}
Available states: {', '.join(available_states)}

Your goal is to follow the script for the current state, engage the user politely, and determine if they are a good fit for Proactiv's plastic card products (referral cards, loyalty cards, gift cards, MOT reminders, appointment cards). 

After your response, suggest the next state by ending with:
<next_state>NEXT_STATE_NAME</next_state>

Choose one of the valid next states from the available states list.
"""
                    conversation.add_message(Message(author="system", content=system_prompt))
                    
                    # Add conversation history manually
                    for turn in self.conversation_history:
                        conversation.add_message(Message(
                            author="user" if turn["role"] == "user" else "assistant",
                            content=turn["content"]
                        ))
                
                # Run LLM inference
                try:
                    # Determine what arguments to include
                    llm_kwargs = {"conversation": conversation}
                    
                    if self.use_memory and self.memory_config:
                        llm_kwargs["memory_config"] = self.memory_config
                    
                    if self.use_tools and self.tools:
                        llm_kwargs["tools"] = self.tools
                    
                    # Call LLM with appropriate arguments
                    llm_response = self.nova.run_llm(**llm_kwargs)
                    response_text = llm_response.message
                    
                    # Execute tool calls if any
                    if self.use_tools and hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
                        logger.info(f"Executing {len(llm_response.tool_calls)} tool calls")
                        self.nova.execute_tool_calls(llm_response)
                    
                    # Extract next state suggestion using enhanced script manager
                    suggested_state = self.script_manager.extract_next_state(response_text)
                    if not suggested_state:
                        suggested_state = current_state
                    
                    # Add LLM response to context
                    self.nova.add_llm_response_to_context(llm_response)
                    
                except Exception as e:
                    logger.error(f"Error using LLM: {e}", exc_info=True)
                    response_text = "I'm having trouble processing that right now."
                    suggested_state = current_state
            else:
                # Fallback responses if LLM not available
                responses = [
                    "I understand. Can you tell me more?",
                    "That's interesting. Let's continue this conversation.",
                    "I see what you mean. Is there anything else you'd like to discuss?",
                    "Thanks for sharing that. How else can I help you today?",
                    "I appreciate your input. Let's move forward."
                ]
                response_text = random.choice(responses)
                suggested_state = current_state
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Add to context manually
            datapoint = ContextDatapoint(
                source=ContextSource_Assistant(),
                content=response_text
            )
            self.nova.add_datapoint_to_context(datapoint)
            
            # Update state if changed
            if suggested_state != current_state:
                logger.info(f"State changing from {current_state} to {suggested_state}")
                self.current_state = suggested_state
                self.event_system.fire("state_changed", suggested_state)
            
            # Generate speech for the response
            self._generate_tts_response(response_text)
            
            # Fire event for LLM response
            llm_response_dict = {
                "response": response_text,
                "next_state": suggested_state
            }
            self.event_system.fire("llm_response_received", llm_response_dict)
            
        except Exception as e:
            logger.error(f"Error processing LLM turn: {e}", exc_info=True)
            # Generate a fallback response
            self._generate_tts_response("I'm having trouble processing that. Let me try again.")
        finally:
            self.processing_input = False
    
    def _generate_tts_response(self, text: str):
        """Generate a TTS response and play it."""
        try:
            logger.info(f"Generating TTS for: {text}")
            
            if self.use_tts:
                try:
                    # Use the Nova API to run TTS
                    audio_data = self.nova.run_tts(text=text, stream=False)
                    
                    # Play the audio 
                    logger.info("Playing TTS response")
                    self.nova.play_audio(audio_data)
                    
                    # Wait for audio playback to finish
                    self.nova.wait_for_audio_playback_end()
                except Exception as tts_error:
                    logger.error(f"TTS generation failed: {tts_error}")
                    logger.warning("TTS failed - no audio will be played")
            else:
                logger.info("TTS disabled - printing response instead")
                print(f"ASSISTANT: {text}")
                
            # Fire event when TTS is completed
            self.event_system.fire("tts_completed", None)
            
        except Exception as e:
            logger.error(f"Error generating TTS response: {e}", exc_info=True)
    
    def _handle_llm_response(self, llm_response: Dict[str, Any]):
        """Handle the LLM response event."""
        logger.debug(f"LLM response received: {llm_response.get('response', '')[:50]}...")
        # Additional processing if needed
    
    def _handle_tts_completed(self, _):
        """Handle the TTS completed event."""
        logger.debug("TTS playback completed")
        # Additional processing if needed
    
    def _handle_state_changed(self, new_state: str):
        """Handle the state changed event."""
        logger.debug(f"State changed to: {new_state}")
        
        # Update the call state machine
        try:
            # Try to convert string to ScriptState enum
            script_state = ScriptState(new_state)
            valid_transition = self.call_state_machine.validate_and_set_state(new_state)
            
            if not valid_transition:
                logger.warning(f"Invalid state transition to {new_state}. Keeping current state {self.call_state_machine.state.value}")
                return
                
            logger.info(f"State machine updated to {script_state.value}")
        except ValueError:
            logger.warning(f"Unknown script state: {new_state}")
        
        # Check if there's a prompt for the new state
        state_prompt = self.script_manager.get_prompt_for_state(new_state)
        if state_prompt:
            logger.info(f"Processing prompt for new state: {new_state}")
            # Queue the prompt for processing
            self.llm_queue.put(state_prompt)
    
    def _cleanup(self):
        """Cleanup resources when shutting down."""
        logger.info("Cleaning up resources")
        
        # Stop the transcriptor
        logger.info("Stopping transcriptor")
        try:
            # Add a small delay to ensure all threads are ready to be stopped
            time.sleep(0.5)
            self.nova.stop_transcriptor()
            logger.info("Transcriptor stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping transcriptor: {e}", exc_info=True)
        
        # Wait for any pending LLM processing to complete
        if self.llm_thread and self.llm_thread.is_alive():
            logger.info("Waiting for LLM processing to complete")
            self.llm_thread.join(timeout=5.0)
        
        # Log final session stats
        start_time = self.context_manager.get_context("start_time")
        if start_time:
            duration = time.time() - start_time
            logger.info(f"Session duration: {duration:.2f} seconds")
            logger.info(f"Final state: {self.current_state}")
            logger.info(f"Conversation turns: {len(self.conversation_history) // 2}")

def run_server():
    """Run the Nova Server with environment-based configuration."""
    # Get configuration from environment variables
    use_llm = os.environ.get("USE_LLM", "true").lower() == "true"
    use_tts = os.environ.get("USE_TTS", "true").lower() == "true"
    use_memory = os.environ.get("USE_MEMORY", "true").lower() == "true"
    use_tools = os.environ.get("USE_TOOLS", "true").lower() == "true"
    script_mode = os.environ.get("SCRIPT_MODE", "standard")
    
    try:
        # Create and run the server
        server = NovaServer(
            use_llm=use_llm,
            use_tts=use_tts,
            use_memory=use_memory,
            use_tools=use_tools
        )
        
        # Initialize script for telemarketer mode if needed
        if script_mode == "telemarketer":
            logger.info("Running in telemarketer script mode")
            # Load the telemarketer script explicitly
            if not server.script_manager.load_script("telemarketer_script"):
                logger.warning("Failed to load telemarketer script, using default script")
            
            # Set initial state to START instead of GREETING
            server.current_state = ScriptState.START.value
            server.call_state_machine.state = ScriptState.START
            
            logger.info(f"Starting with initial state: {server.current_state}")
        
        # Start the server session
        server.start_session()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.exception(f"Error running Nova Server: {e}")

if __name__ == "__main__":
    run_server()