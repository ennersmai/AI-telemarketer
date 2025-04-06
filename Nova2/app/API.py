"""
Description: Acts as the base API for the main API and the tool API.
"""

import logging
from typing import Union, Optional, Any, AsyncGenerator

import uvicorn
import asyncio
import queue
import threading

# --- Nova imports ---
from Nova2.app.tts_manager import *
from Nova2.app.llm_manager import *
from Nova2.app.audio_manager import *
from Nova2.app.transcriptor import VoiceAnalysis
from Nova2.app.transcriptor_data import TranscriptorConditioning
from Nova2.app.context_manager import ContextManager
from Nova2.app.context_data import *
from Nova2.app.inference_engines import *
from Nova2.app.security_manager import *
# --- End Nova imports ---

# --- Adapter Class ---
# NOTE: Consider moving this to a separate utility file later
class AsyncGeneratorAdapter:
    def __init__(self, agen: AsyncGenerator[Any, None], loop: asyncio.AbstractEventLoop):
        if not asyncio.iscoroutinefunction(agen.__anext__):
             # Handle cases where it might still be a sync generator if testing both ways
             raise TypeError("AsyncGeneratorAdapter requires an AsyncGenerator")
             
        self._agen = agen
        self._loop = loop
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._consumer_task = None
        self._started = False
        self._exception = None
        self._lock = threading.Lock()

    async def _consume_agen(self):
        """The async task that consumes the async generator."""
        try:
            async for item in self._agen:
                if self._stop_event.is_set():
                    logger.debug("[AsyncGeneratorAdapter] Stop event set during iteration.")
                    break
                # logger.debug(f"[AsyncGeneratorAdapter] Putting item on queue: {type(item)}")
                self._queue.put(item)
            # Put a sentinel value to indicate the generator is exhausted
            logger.debug("[AsyncGeneratorAdapter] Generator exhausted, putting sentinel.")
            self._queue.put(StopIteration)
        except asyncio.CancelledError:
            logger.info("[AsyncGeneratorAdapter] Consumer task cancelled.")
            self._queue.put(StopIteration) # Ensure consumer stops
        except Exception as e:
             logger.error(f"[AsyncGeneratorAdapter] Exception during async generator consumption: {e}", exc_info=True)
             self._queue.put(e) # Put exception onto queue
             self._exception = e # Also store it

    def _start_consumer(self):
        """Schedules the consumer task on the provided event loop."""
        with self._lock:
             if not self._started:
                 logger.info("[AsyncGeneratorAdapter] Scheduling consumer task on the event loop...")
                 # Schedule the task directly on the loop passed during init
                 if self._loop and self._loop.is_running():
                      self._consumer_task = self._loop.create_task(self._consume_agen())
                      self._started = True
                      logger.info("[AsyncGeneratorAdapter] Consumer task scheduled.")
                 else:
                      logger.error("[AsyncGeneratorAdapter] Cannot schedule task: Provided loop is not running.")
                      # Raise error or handle? Store exception?
                      self._exception = RuntimeError("Event loop for adapter is not running")
                      self._queue.put(self._exception) # Put error for __next__

    def __iter__(self):
        # Ensure the consumer is started when iteration begins
        if not self._started:
             self._start_consumer()
        return self

    def __next__(self):
        """Synchronous next method, gets item from the queue."""
        # Ensure consumer is started if iterator created but next never called initially
        if not self._started:
             self._start_consumer()
             
        # Check for exceptions raised by the consumer task first
        # Access self._exception safely if needed (maybe lock?)
        if self._exception:
             # Log or handle stored exception
             # This path might be less reliable than getting Exception from queue
             pass
             
        try:
            # Block until an item is available or StopIteration sentinel is received
            # logger.debug("[AsyncGeneratorAdapter] Waiting for item from queue...")
            item = self._queue.get(block=True, timeout=None) # Blocking get
            # logger.debug(f"[AsyncGeneratorAdapter] Got item from queue: {type(item)}")
            if item is StopIteration:
                logger.info("[AsyncGeneratorAdapter] Received StopIteration sentinel.")
                # Wait for the task to actually finish if needed
                # if self._consumer_task and not self._consumer_task.done():
                #    try:
                #       # This is complex as we are in a sync method
                #       # Best effort: signal stop if not already set
                #       self._stop_event.set()
                #    except Exception as e:
                #        logger.error(f"Error ensuring task stop: {e}")
                raise StopIteration
            elif isinstance(item, Exception):
                 logger.error(f"[AsyncGeneratorAdapter] Received Exception from queue: {item}")
                 raise item # Re-raise exception from consumer
            return item
        except queue.Empty:
            # This shouldn't happen with block=True unless queue broken
            logger.error("[AsyncGeneratorAdapter] Queue empty unexpectedly in blocking __next__.")
            raise StopIteration
        except Exception as e:
             logger.error(f"[AsyncGeneratorAdapter] Error in __next__: {e}", exc_info=True)
             raise

    def stop(self):
        """Signals the consuming task to stop."""
        logger.info("[AsyncGeneratorAdapter] Stop requested.")
        self._stop_event.set()
        # Try to cancel the asyncio task if running
        if self._consumer_task and not self._consumer_task.done():
             logger.debug("[AsyncGeneratorAdapter] Requesting task cancellation via loop.")
             # Schedule cancellation from the loop's thread (still safe)
             self._loop.call_soon_threadsafe(self._consumer_task.cancel)
        # Put sentinel to ensure __next__ stops blocking if waiting
        # Do this unconditionally, as task might finish before cancel registers
        self._queue.put(StopIteration)
        logger.debug("[AsyncGeneratorAdapter] Stop request processed.")

# --- End Adapter Class ---

class NovaAPI:
    def __init__(self) -> None:
        """
        Novas API.
        """
        self._tts = TTSManager()
        print(f"[DEBUG][NovaAPI] __init__ created TTSManager: {bool(self._tts)}")
        self._llm = LLMManager()
        self._stt = VoiceAnalysis()

        self._context = ContextManager()
        self._context_data = ContextManager()
        self._player = AudioPlayer()
        self._tools = ToolManager()
        self._security = SecretsManager()

        logging.getLogger().setLevel(logging.CRITICAL)

    def configure_transcriptor(self, conditioning: TranscriptorConditioning) -> None:
        """
        Configure the transcriptor.
        """
        print("[DEBUG][NovaAPI] configure_transcriptor called.")
        if not hasattr(self, '_stt') or self._stt is None:
            print("[ERROR][NovaAPI] _stt object not found in configure_transcriptor!")
            return
        try:
            self._stt.configure(conditioning=conditioning)
            print("[DEBUG][NovaAPI] Called self._stt.configure().")
        except Exception as e:
            print(f"[ERROR][NovaAPI] Error calling self._stt.configure(): {e}")

    def configure_llm(self, inference_engine: InferenceEngineBaseLLM, conditioning: LLMConditioning) -> None:
        """
        Configure the LLM system.
        """
        self._llm.configure(inference_engine=inference_engine, conditioning=conditioning)

    def configure_tts(self, inference_engine: InferenceEngineBaseTTS, conditioning: TTSConditioning) -> None:
        """
        Configure the TTS engine.
        """
        print("\n!!!!!!!!!!!!! INSIDE NovaAPI.configure_tts !!!!!!!!!!!!!\n")
        print(f"[DEBUG][NovaAPI] configure_tts called. Engine: {type(inference_engine)}, Conditioning: {type(conditioning)}")
        self._tts.configure(
            inference_engine=inference_engine,
            conditioning=conditioning
        )
        print(f"[DEBUG][NovaAPI] configure_tts finished calling self._tts.configure.")
        print("\n!!!!!!!!!!!!! LEAVING NovaAPI.configure_tts !!!!!!!!!!!!!\n")

    def apply_config_all(self) -> None:
        """
        Updates the configuration of the transcriptor, LLM and TTS systems. Also loads the chosen models into memory.
        """
        self._tts.apply_config()
        self._llm.apply_config()
        self._stt.apply_config()

    def apply_config_llm(self) -> None:
        """
        Updates the configuration of the LLM system. Also loads the chosen models into memory.
        """
        self._llm.apply_config()

    def apply_config_tts(self) -> None:
        """
        Apply the TTS configuration.
        """
        print("[DEBUG][NovaAPI] apply_config_tts called.")
        self._tts.apply_config()
        print("[DEBUG][NovaAPI] apply_config_tts finished calling self._tts.apply_config.")

    def apply_config_transcriptor(self) -> None:
        """
        Updates the configuration of the trabscriptor system. Also loads the chosen models into memory.
        """
        print("[DEBUG][NovaAPI] apply_config_transcriptor called.")
        if not hasattr(self, '_stt') or self._stt is None:
            print("[ERROR][NovaAPI] _stt object not found in apply_config_transcriptor!")
            return
        try:
            self._stt.apply_config()
            print("[DEBUG][NovaAPI] Called self._stt.apply_config().")
        except Exception as e:
            import traceback
            print(f"[ERROR][NovaAPI] Error calling self._stt.apply_config(): {e}")
            print(f"[ERROR][NovaAPI] Traceback: {traceback.format_exc()}")
            raise

    def load_tools(self, load_internal_tools: bool = True, **kwargs) -> List[LLMTool]:
        """
        Load all tools in the tool folder into memory and make them ready for calling.

        Arguments:
            load_internal_tools (bool): Wether the set of internal tools should be loaded. If set to false, the LLM loses access to some core functionality like renaming voices in the database or creating new memories.
            include (List[string]): Which tools should be loaded. Incompatible with "exclude".
            exclude (List[string]): Which tools should not be loaded. Incompatible with "include".

        Returns:
            List[LLMTool]: A list of loaded tools that can be parsed when running the LLM to give it access to these tools.
        """
        return self._tools.load_tools(load_internal=load_internal_tools, **kwargs)
    
    def execute_tool_calls(self, llm_response: LLMResponse) -> None:
        """
        Execute the tools that were called by the LLM.

        Arguments:
            tool_calls (List[LLMToolCall]): The tool calls from the LLM that should be executed.
        """
        self._tools.execute_tool_call(tool_calls=llm_response.tool_calls)

    def run_llm(self, conversation: Conversation, memory_config: MemoryConfig = None, tools: List[LLMTool] = None, instruction: str = "") -> LLMResponse:
        """
        Run inference on the LLM.

        Arguments:
            conversation (Conversation): A conversation to use. Can be retrieved from context.
            memory_config (MemoryConfig): How should memories be retieved? If none is provided, no memories will be retrieved.
            tools (list[LLMTool]): A list of tools the LLM can access.
            instruction (str): An additional instruction to give to the LLM.

        Returns:
            LLMResponse: A response object containing all relevant information about what the LLM has responded.
        """
        return self._llm.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config, instruction=instruction)

    def run_tts(self, text: str, stream: bool = False) -> Union[AudioData, StreamingAudioData]:
        """
        Run inference on the TTS.

        Arguments:
            text (str): The text that should be turned into speech.
            stream (bool): Whether to stream the audio data.

        Returns:
            Union[AudioData, StreamingAudioData]: The resulting audio data that can be played by the audio player,
                                               or a streaming audio iterator if stream=True.
        """
        return self._tts.run_inference(text=text, stream=stream)

    def start_transcriptor(self, context_id: str = "default") -> Optional[ContextGenerator]:
        """
        Start the transcriptor. Returns a ContextGenerator wrapping an adapter.
        The transcriptor will start to listen to the microphone audio.
        """
        print(f"[DEBUG][NovaAPI] start_transcriptor called for context_id: {context_id}")
        if not hasattr(self, '_stt') or self._stt is None:
            print("[ERROR][NovaAPI] _stt object not found in start_transcriptor!")
            return None
        try:
            # Get the async generator from VoiceAnalysis
            # Need to await the async start method now
            # This requires start_transcriptor itself to be async or run within an event loop
            # Assuming it's called from an async context (like run_local_test)
            
            # ***** MAJOR CHANGE NEEDED HERE *****
            # We cannot directly call/await self._stt.start() from this synchronous method.
            # This requires rethinking how start_transcriptor is called or how the server runs.
            
            # --- TEMPORARY WORKAROUND (May block or fail depending on caller context) ---
            # Option A: Use asyncio.run (BAD if called from existing async context)
            # try:
            #     stt_async_generator = asyncio.run(self._stt.start()) 
            # except RuntimeError as e:
            #     logger.error(f"Cannot use asyncio.run: {e}")
            #     return None
                
            # Option B: Assume called from within loop (like run_local_test)
            # This path is complex. The calling code (run_local_test) needs to await self._stt.start()
            # and then pass the resulting async generator to this (or a modified) function.
            
            # Let's assume for NOW that the caller handles awaiting self._stt.start()
            # and passes the async generator *object* (not awaited result) here.
            # This API design is now awkward.
            # We will modify this to *receive* the started async generator.
            
            # THIS METHOD SIGNATURE/PURPOSE NEEDS REVISION
            # For now, let's comment out the problematic part and assume 
            # the adapter creation happens elsewhere or this method isn't used directly.
            
            logger.warning("[NovaAPI] start_transcriptor needs refactoring to handle async VoiceAnalysis.start()")
            return None # Cannot proceed with current synchronous structure
            
            # --- The following logic would run if stt_async_generator was obtained correctly --- 
            # stt_async_generator = self._stt.start() # Hypothetical call returning async gen
            # print("[DEBUG][NovaAPI] Called self._stt.start(), received async generator object.")
            # try:
            #      loop = asyncio.get_running_loop()
            # except RuntimeError:
            #      logger.error("[NovaAPI] No running asyncio event loop found in start_transcriptor!")
            #      return None
            # adapter = AsyncGeneratorAdapter(stt_async_generator, loop)
            # print("[DEBUG][NovaAPI] Created AsyncGeneratorAdapter.")
            # context_gen = ContextGenerator(adapter, context_id=context_id)
            # print("[DEBUG][NovaAPI] Wrapped adapter in ContextGenerator.")
            # return context_gen
            # --- End hypothetical logic --- 

        except Exception as e:
            print(f"[ERROR][NovaAPI] Error in start_transcriptor: {e}")
            logger.error(f"Error in start_transcriptor: {e}", exc_info=True) # Add logger
            return None

    def start_transcriptor_async(self, context_id: str):
        """Start the transcriptor in a synchronous manner and bind it to the given context."""
        logger.debug(f"[NovaAPI] start_transcriptor_async called for context_id: {context_id}")
        try:
            # Use the synchronous version instead of async
            stt_generator = self._stt.start_sync()
            
            # No need for AsyncGeneratorAdapter - the generator is already synchronous
            if stt_generator:
                logger.info(f"[NovaAPI] Created synchronous STT generator for context {context_id}")
                return ContextGenerator(context_id=context_id, generator=stt_generator, callback=self._stt_callback)
            else:
                logger.error(f"[NovaAPI] Failed to get STT generator from start_sync()")
                return None
        except Exception as e:
            logger.error(f"[NovaAPI] Error in start_transcriptor_async: {e}", exc_info=True)
            return None

    def stop_transcriptor(self, context_id: str = "default"):
        """Stop the transcriptor associated with a context_id."""
        print(f"[DEBUG][NovaAPI] stop_transcriptor called for context_id: {context_id}")
        if hasattr(self, '_stt') and hasattr(self._stt, 'close'):
            # Note: Current VoiceAnalysis.close isn't context specific.
            # This will stop the single instance.
            # For multi-call, STT needs context management.
            self._stt.close()
            print(f"[DEBUG][NovaAPI] Called self._stt.close() for context {context_id}")
        else:
             print(f"[WARN][NovaAPI] _stt object or close method not found for context {context_id}")

    def bind_context_source(self, source: ContextGenerator) -> None:
        """
        Bind a context source. The data of a context source will only be recorded after beeing bound.
        """
        self._context.record_data(source)

    def get_context(self) -> Context:
        """
        Get the current context.
        """
        return self._context_data.get_context_data()
    
    def set_context(self, context: Context) -> None:
        """
        Overwrites the stored context data.
        """
        self._context_data._overwrite_context(context.data_points)
    
    def set_ctx_limit(self, ctx_limit: int) -> None:
        """
        Limit how many datapoints will be stored in context. This does not include memory.
        Setting it to 0 will impose no limit, but the context will surpass the LLMs context window at some point.
        Limit is 25 by default.
        """
        self._context_data.ctx_limit = ctx_limit

    def add_to_context(self, name: str, content: str, id: str) -> None:
        """
        Add a response from the tool to the context.

        Arguments:
            name (str): The name of the tool. Should match the name given in metadata.json.
            content (str): The message that should be added to the context
        """
        dp = ContextDatapoint(
            source=ContextSource_ToolResponse(
                name=name,
                id=id
            ),
            content=content
        )

        ContextManager().add_to_context(datapoint=dp)
    
    def add_llm_response_to_context(self, response: LLMResponse) -> None:
        """
        Add LLMResponse to the context.
        """
        if len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                self._context_data.add_to_context(
                    ContextDatapoint(
                        source=ContextSource_Assistant(),
                        content=f"Called tool \"{tool_call.name}\""
                    ))
        else:
            self._context_data.add_to_context(
                ContextDatapoint(
                    source=ContextSource_Assistant(),
                    content=response.message
                ))

    def play_audio(self, audio_data: AudioData) -> None:
        """
        Use the built in audio player to play audio. Only accepts an AudioData object.
        """
        self._player.play_audio(audio_data)

    def wait_for_audio_playback_end(self) -> None:
        """
        Halts the code execution until the audio player is done playing the current audio.
        """
        while self._player.is_playing():
            time.sleep(0.1)

    def is_playing_audio(self) -> bool:
        """
        Checks wether the audio player is currently playing any audio.
        """
        return self._player.is_playing()
    
    def huggingface_login(self, overwrite: bool = False, token: str = "") -> None:
        """
        Attempt to log into huggingface which is required to access restricted repos.
        Raises an exception if the login fails.
        
        Arguments:
            overwrite (bool): If true, "token" will overwrite the value stored in the database. If false, the database will remain unchanged and "token" will be used to attempt a login, if provided.
            token (str): If provided, this token will be used to log in.
        """
        self._security.huggingface_login(overwrite=overwrite, token=token)

    def edit_secret(self, name: Secrets, value: str) -> None:
        """
        Edit a secret, like an API key in the database. The value will be encrypted before it is stored.

        Arguments:
            name (Secrets): Which of the secrets to edit.
            value (str): The new value of the secret.
        """
        self._security.edit_secret(name=name, key=value)