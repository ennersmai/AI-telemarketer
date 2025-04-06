"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""

from threading import Thread
import time
from pathlib import Path
import json
import asyncio
from typing import Dict, Tuple

import torch

from .context_data import *
from .database_manager import VoiceDatabaseManager
from .transcriptor_data import Word

class ContextManager:
    source_list = ContextGeneratorList()
    context_data = []
    _voice_callbacks: Dict[str, Tuple[callable, asyncio.AbstractEventLoop]] = {}
    
    def __init__(self) -> None:
        """
        Prepares context data provided by a listener and stores them in the context file.
        """
        self._voice_database_manager = VoiceDatabaseManager()

        self._context_file = Path(__file__).parent.parent / "data" / "context.json"
        ContextManager.context_data = self._prepare_context_data()

        self.ctx_limit = 25

        self._context_recorder = Thread(target=self._record_context, daemon=True)
        self._context_recorder.start()

    def record_data(self, source: ContextGenerator) -> None:
        """
        Begins to listen to the source and record the data.
        """
        ContextManager.source_list.add(context_source=source)

    def register_voice_callback(self, context_id: str, func: callable, loop: asyncio.AbstractEventLoop):
        """Register an async callback for voice context data."""
        print(f"[DEBUG][ContextManager] Registering voice callback for context_id: {context_id}")
        self._voice_callbacks[context_id] = (func, loop)

    def unregister_voice_callback(self, context_id: str):
        """Unregister a voice callback."""
        print(f"[DEBUG][ContextManager] Unregistering voice callback for context_id: {context_id}")
        self._voice_callbacks.pop(context_id, None)

    def _record_context(self) -> None:
        """
        Stores the context of all bound context sources.
        """
        while True:
            result = ContextManager.source_list.get_next()
            if not result:
                time.sleep(0.1)
                continue

            datapoint, context_id = result
            print(f"[DEBUG][ContextManager._record_context] Received from get_next(): context_id={context_id}, datapoint_type={type(datapoint).__name__}", flush=True)
            if isinstance(datapoint, ContextDatapoint):
                 print(f"    Content Snippet: '{str(datapoint.content)[:30]}...'", flush=True)

            if not datapoint:
                continue

            self.add_to_context(datapoint=datapoint, context_id=context_id)

            time.sleep(0.01)

    def add_to_context(self, datapoint: ContextDatapoint, context_id: str = "default") -> None:
        """
        Adds content to the context.json file and triggers callbacks.

        Arguments:
            datapoint (ContextDatapoint): The datapoint that will be added to the context.
            context_id (str): Identifier for the context (e.g., call_sid).
        """
        ContextManager.context_data.append(datapoint.to_dict())

        if self.ctx_limit > 0:
            ContextManager.context_data = ContextManager.context_data[-self.ctx_limit:]

        self.save_context_data()
        
        if isinstance(datapoint.source, ContextSource_Voice):
            if context_id in self._voice_callbacks:
                callback_func, loop = self._voice_callbacks[context_id]
                text_content = datapoint.content
                print(f"[DEBUG][ContextManager] Scheduling voice callback for {context_id} with text: '{text_content[:30]}...'")
                coro = callback_func(text_content, context_id)
                asyncio.run_coroutine_threadsafe(coro, loop)

    def _overwrite_context(self, context: List[ContextDatapoint]) -> None:
        """
        Overwrites the entire context. Use with caution.

        Arguments:
            context (List[ContextDatapoint]): The data the context will be overwritten with.
        """
        ContextManager.context_data = []

        for datapoint in context:
            self.add_to_context(datapoint=datapoint)

    def save_context_data(self) -> None:
        """
        Saves the context data to the context.json file.
        """
        with open(self._context_file, 'w') as file:
            json.dump(ContextManager.context_data, file, indent=4)

    def get_context_data(self) -> Context:
        """
        Reconstructs a Context object from the stored context.

        Returns:
            Context: The context data stored in memory.
        """
        sources = ContextSourceBase.get_all_sources()

        datapoints = []

        for datapoint in ContextManager.context_data:
            source_instance = None
            # Find the correct source
            for source in sources:
                if source.__name__ == datapoint["source"]["type"] and source.__name__ != "ContextSourceBase":
                    if "metadata" in datapoint["source"]:
                        metadata = datapoint["source"]["metadata"]
                        source_instance = source(**metadata)
                    else:
                        source_instance = source()
                    break
            
            if not source_instance:
                dp = datapoint["source"]["type"]
                raise Exception(f"Got unknown context source {dp}")

            datapoint_instance = ContextDatapoint(
                source=source_instance,
                content=datapoint["content"]
                )
            
            datapoint_instance.timestamp = datapoint["timestamp"]
            datapoints.append(datapoint_instance)

        return Context(datapoints)
    
    def rename_voice(self, old_name: str, new_name: str) -> None:
        """
        Renames a voice in the context.

        Arguments:
            old_name (str): The current name of the voice.
            new_name (str): What the voice should be renamed to.
        """
        context = []

        for datapoint in self.get_context_data().data_points:
            if type(datapoint.source) == ContextSource_Voice:
                if datapoint.source.speaker == old_name:
                    dp = ContextDatapoint(
                        source=ContextSource_Voice(
                            speaker=new_name,
                        ),
                        content=datapoint.content
                    )
                    context.append(dp)
                else:
                    context.append(datapoint)

        self._overwrite_context(context=context)

    def _prepare_context_data(self) -> List[dict]:
        """
        Creates a context.json file if it does not exist. Returns the contents of the context.json file.
        
        Returns:
            list[dict]: The contents of the context.json file.
        """
        if not self._context_file.exists():
            with open(self._context_file, 'w') as file:
                json.dump([], file)

        with open(self._context_file, 'r') as file:
            context_data = json.load(file)

        return context_data

    def _word_array_to_string(self, word_array: list[Word]) -> str:
        text = ""
        for word in word_array:
            text += word.text
        return text
    
    def _take_average_embedding(self, embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        return torch.mean(torch.stack(embeddings), dim=0)