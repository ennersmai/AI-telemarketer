"""
Event System for Nova2 Project

Provides both a functional event system API and an object-oriented EventSystem class.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global event registry for the functional API
_events: Dict[str, List[Callable]] = {}

class EventListener:
    """
    Event listener class for the object-oriented event system.
    """
    
    def __init__(self, callback: Callable, once: bool = False):
        """
        Initialize an event listener.
        
        Args:
            callback: The function to call when the event is triggered
            once: If True, the listener will be removed after the first call
        """
        self.callback = callback
        self.once = once
        self.active = True
    
    def __call__(self, *args, **kwargs):
        """Make the listener callable directly."""
        if self.active:
            if self.once:
                self.active = False
            return self.callback(*args, **kwargs)


class EventSystem:
    """
    Object-oriented event system implementation that can be instantiated
    for scoped event handling.
    """
    
    def __init__(self):
        """Initialize a new event system instance."""
        self._events: Dict[str, List[EventListener]] = {}
    
    def register(self, event_name: str, listener: EventListener) -> None:
        """
        Register a listener for an event.
        
        Args:
            event_name: The name of the event
            listener: The EventListener to register
        """
        if event_name not in self._events:
            self._events[event_name] = []
        
        self._events[event_name].append(listener)
        logger.debug(f"Registered listener for event: {event_name}")
    
    def unregister(self, event_name: str, listener: EventListener) -> None:
        """
        Unregister a listener from an event.
        
        Args:
            event_name: The name of the event
            listener: The EventListener to unregister
        """
        if event_name in self._events and listener in self._events[event_name]:
            self._events[event_name].remove(listener)
            logger.debug(f"Unregistered listener from event: {event_name}")
    
    def fire(self, event_name: str, event_data: Any = None) -> None:
        """
        Fire an event, triggering all registered listeners.
        
        Args:
            event_name: The name of the event to fire
            event_data: Optional data to pass to listeners
        """
        if event_name not in self._events:
            logger.debug(f"No listeners registered for event: {event_name}")
            return
        
        # Create a copy of the listeners to allow for listeners that remove themselves
        listeners = self._events[event_name].copy()
        
        for listener in listeners:
            try:
                listener(event_data)
                
                # Remove one-time listeners after firing
                if listener.once and listener in self._events[event_name]:
                    self._events[event_name].remove(listener)
            except Exception as e:
                logger.error(f"Error in event listener for {event_name}: {e}", exc_info=True)
    
    def clear(self, event_name: Optional[str] = None) -> None:
        """
        Clear all listeners for an event or all events.
        
        Args:
            event_name: The name of the event to clear, or None to clear all events
        """
        if event_name:
            if event_name in self._events:
                self._events[event_name] = []
                logger.debug(f"Cleared all listeners for event: {event_name}")
        else:
            self._events = {}
            logger.debug("Cleared all event listeners")


# Keep the original functional API for backward compatibility

def event_exists_error_handling(func: Callable):
    """
    Raises an exception if something tries to interact with an event that does not exist.
    """
    def wrapper(event_name: str, *args, **kwargs):
        if event_name not in _events:
            raise Exception(f"Event {event_name} does not exist.")
        return func(event_name, *args, **kwargs)
    return wrapper

def define_event(event_name: str) -> None:
    """
    Defines a new event.

    Arguments:
        event_name (str): The name of the new event.
    """
    if event_name in _events:
        raise Exception(f"Event {event_name} already exists.")

    _events[event_name] = []

@event_exists_error_handling
def subscribe(event_name: str, callback: Callable) -> None:
    """
    Subscribe to an event.

    Arguments:
        event_name (str): The event that should be subscribed to.
        callback (callable): The callable that will be called when the event is triggered.
    """
    _events[event_name].append(callback)

@event_exists_error_handling
def unsubscribe(event_name: str, callback: Callable) -> None:
    """
    Unsubscribe from an event.

    Arguments:
        event_name (str): The event that should be unsubscribed from.
        callback (callable): The callable that will no longer be called when the event is triggered.
    """
    _events[event_name].remove(callback)

@event_exists_error_handling
def is_subscribed(event_name: str, callback: Callable) -> bool:
    """
    Checks if a callable is subscribed to an event.

    Arguments:
        event_name (str): The event that should be checked.
        callback (callable): The callable that should be checked.

    Returns:
        bool: True if the callable is subscribed to the event.
    """
    return callback in _events[event_name]

@event_exists_error_handling
def trigger_event(event_name: str, *args, **kwargs) -> None:
    """
    Triggers an event. All subscribed callables will be called.

    Arguments:
        event_name (str): The event that should be triggered.
        args, kwargs: Additional arguments that should be passed to the callables.
    """
    for callback in _events[event_name]:
        try:
            callback(*args, **kwargs)
        except Exception as e:
            raise Exception(f"Unable to call {callback} from event {event_name}. Error: {e}")

def event_exists(event_name: str) -> bool:
    """
    Checks if an event exists.

    Arguments:
        event_name (str): Check if an event with this name exists.

    Returns:
        bool: True if the event exists.
    """
    return event_name in _events