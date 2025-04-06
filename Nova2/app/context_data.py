"""
Description: Holds all data related to context data.
"""

from typing import Generator, List, Optional, Tuple
from datetime import datetime
from queue import Queue
from enum import Enum
from threading import Thread
import time
import threading

from .llm_data import Conversation, Message

class ContextSourceBase:
    def __init__(self):
        pass
    
    @classmethod
    def get_all_sources(cls) -> List[type]:
        return cls.__subclasses__()

class ContextSource_Voice(ContextSourceBase):
    def __init__(
                self,
                speaker: str
                ) -> None:
        self.speaker = speaker

class ContextSource_User(ContextSourceBase):
    def __init__(self) -> None:
        pass

class ContextSource_Assistant(ContextSourceBase):
    def __init__(self) -> None:
        pass

class ContextSource_ToolResponse(ContextSourceBase):
    def __init__(
                self,
                name: str,
                id: str
                ) -> None:
        self.name = name
        self.id = id

class ContextSource_System(ContextSourceBase):
    def __init__(self) -> None:
        pass

class ContextDatapoint:
    def __init__(
                self,
                source: ContextSourceBase,
                content: str,
                ) -> None:
        """
        This class holds a singular datapoint in the context.
        """
        self.source = source
        self.content = content
        self.timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        """
        Returns the contents formatted to a dictionary so it can be serialized to json.
        """
        # Check if the source contains metadata
        if bool(self.source.__dict__):
            return {
                "source": {
                    "type": self.source.__class__.__name__,
                    "metadata": self.source.__dict__
                },
                "content": self.content,
                "timestamp": self.timestamp
            }
        else:
            return {
                "source": {
                    "type": self.source.__class__.__name__
                },
                "content": self.content,
                "timestamp": self.timestamp
            }

class Context:
    def __init__(
                self,
                data_points: list[ContextDatapoint]
                ) -> None:
        """
        This class stores context which is a list of datapoints all with source, content and timestamp.
        """
        self.data_points = data_points

    def to_conversation(self) -> Conversation:
        """
        Get the context as type Conversation that can be parsed to the LLM.
        """
        messages = []

        # Thank you python that I am not allowed to use a match-case here.
        for datapoint in self.data_points:
            if type(datapoint.source) == ContextSource_Assistant:
                messages.append(
                    Message(
                        author="assistant",
                        content=datapoint.content # Assistant message does not need a timestamp
                ))
            elif type(datapoint.source) == ContextSource_Voice:
                messages.append(
                    Message(
                        author="user",
                        content=f"{datapoint.source.speaker} ({datapoint.timestamp}): {datapoint.content}"
                ))
            elif type(datapoint.source) == ContextSource_ToolResponse:
                messages.append(
                    Message(
                        author="tool",
                        name=datapoint.source.name,
                        tool_call_id=datapoint.source.id,
                        content=f"({datapoint.timestamp}) {datapoint.content}"
                    )
                )
            elif type(datapoint.source) == ContextSource_System:
                messages.append(
                    Message(
                        author="system",
                        content=f"{datapoint.timestamp}: {datapoint.content}"
                    )
                )
            else:
                raise Exception(f"Could not format context to conversation. Unknown source: {type(datapoint.source)}")

        return Conversation(messages)

class ContextGenerator:
    def __init__(self, generator: Generator, context_id: str = "default") -> None:
        """
        A wrapper class for generators that yield context datapoints.

        Arguments:
            generator (Generator): The generator to wrap.
            context_id (str): An identifier for the context this generator belongs to.
        """
        self._generator = generator
        self.context_id = context_id

    def __aiter__(self):
        # Allows async iteration if the underlying generator supports it (or wrapped)
        # For simplicity, we might assume synchronous usage in _record_context
        return self

    async def __anext__(self):
        # For async iteration - might not be needed if _record_context is sync
        try:
            # If _generator is sync, need to run in executor or handle StopIteration
            return next(self._generator) 
        except StopIteration:
            raise StopAsyncIteration

    # Add standard iterator methods if needed for sync iteration
    def __iter__(self):
        return self._generator

    def __next__(self):
        return next(self._generator)

class ContextGeneratorList:
    def __init__(self) -> None:
        self._generators: List[ContextGenerator] = []
        self._lock = threading.Lock()
        self._current_generator_index = 0

    def add(self, context_source: ContextGenerator) -> None:
        with self._lock:
            self._generators.append(context_source)

    def get_next(self) -> Optional[Tuple[ContextDatapoint, str]]:
        """Gets the next datapoint from the available generators in a round-robin fashion."""
        with self._lock:
            if not self._generators:
                return None

            start_index = self._current_generator_index
            while True:
                generator_wrapper = self._generators[self._current_generator_index]
                try:
                    # Use the standard iterator protocol
                    datapoint = next(generator_wrapper)
                    context_id = generator_wrapper.context_id
                    # Move to the next generator for the next call
                    self._current_generator_index = (self._current_generator_index + 1) % len(self._generators)
                    return datapoint, context_id
                except StopIteration:
                    # Generator finished, remove it (optional, depends on desired behavior)
                    # print(f"[DEBUG][ContextGeneratorList] Generator for {generator_wrapper.context_id} finished.")
                    # self._generators.pop(self._current_generator_index)
                    # if not self._generators: return None
                    # # Adjust index if we removed the last element
                    # self._current_generator_index %= len(self._generators) 
                    # For continuous streaming, just move to next without removing
                    pass 
                except Exception as e:
                    print(f"[ERROR][ContextGeneratorList] Error getting data from generator {generator_wrapper.context_id}: {e}")
                    # Optionally remove problematic generator
                    # self._generators.pop(self._current_generator_index)
                    # ... (adjust index) ...
                    pass # Move to next generator
                
                # Move to the next generator index
                self._current_generator_index = (self._current_generator_index + 1) % len(self._generators)
                # If we've looped back to the start without finding data, return None for now
                if self._current_generator_index == start_index:
                    return None

class ListCommands(Enum):
    ADD = "add"
    REMOVE = "remove"
    GET_NEXT = "get_next"