# Nova2/tools/end_call/end_call.py

# Note: Based on the simpler approach where the tool only signals intent.
# The actual call termination logic is handled in telemarketer_server.py
# after nova.execute_tool_calls identifies this tool was called.

# Imports might not be strictly needed if tool_api is available implicitly 
# and we don't call NovaAPI methods from here directly.
# from tool_api import ToolBaseClass, Nova 

import logging
import sys
from pathlib import Path
import requests 
import os

# Add the base directory to the path to allow importing tool_api
# This assumes the script is run in a context where 'Nova2' is the root for imports, 
# or we need to adjust the path relative to this file's location.
tool_dir = Path(__file__).parent
base_dir = tool_dir.parent.parent # Go up two levels (tools -> Nova2)
sys.path.insert(0, str(base_dir))

from tool_api import ToolBaseClass, Nova

logger = logging.getLogger(__name__)

# Assume ToolBaseClass is implicitly available or handled by the loading mechanism
class Tool(ToolBaseClass):
    """
    Tool for the LLM to signal the intent to end the call.
    Actual termination is handled by the server upon detecting this tool call.
    """
    def on_startup(self) -> None:
        logger.info("End Call tool startup.")

    def on_call(self, farewell_message: str | None = None, **kwargs) -> None:
        """
        Called by the LLM to signal the end of the call.
        
        Args:
            farewell_message (Optional[str]): Custom farewell message.
        """
        logger.info(f"End Call tool called by LLM for call_sid: {self._call_sid} (via tool context)")
        
        # This tool doesn't need to add context itself, as the server 
        # will detect the call and initiate termination.
        # If we needed to pass the farewell message back reliably, 
        # we might add it to context here, but let's try extracting it 
        # from the tool_calls object in the server first.
        pass 