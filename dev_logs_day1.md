# Nova2 Telemarketer System - Development Log Day 1

## Overview

This development log documents the implementation of the Nova2 Telemarketer System - an AI-powered voice calling platform that leverages Twilio's Programmable Voice API for real-time bidirectional audio streaming, combined with Nova2's Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.

The system enables:
- Inbound call handling with automated AI responses
- Outbound call management with queue and rate limiting
- Real-time bidirectional audio streaming
- UK calling regulations compliance
- LLM-driven conversation management
- State tracking for complex conversation flows

## System Architecture

The system consists of the following key components:

1. **Telemarketer Server** (`telemarketer_server.py`): A FastAPI application that serves as the central coordinator, handling Twilio webhooks, WebSocket connections, and API endpoints.

2. **STT Integration** (`stt_integration.py`): Connects to Nova2's STT via WebSockets to process incoming caller audio in real-time.

3. **Streaming Telemarketer** (`stream_telemarketer.py`): Leverages Nova2's TTS capabilities to generate audio responses streamed to callers.

4. **Call State Manager** (`call_state_manager.py`): Maintains conversation state and manages transitions between dialog states.

5. **UK Call Regulations** (`uk_call_regulations.py`): Enforces compliance with UK telemarketing regulations, including time restrictions, frequency limitations, and TPS registry checking.

6. **LLM Integration**: Processes conversation content, determines next actions, and controls when to end conversations.

### Architecture Diagram

```
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│  Twilio Voice   │◄────►│  Telemarketer   │
│     System      │ HTTP │     Server      │
│                 │      │   (FastAPI)     │
└────────┬────────┘      └───┬─────────┬───┘
         │                   │         │
         │                   │         │
         │WebSocket          │         │
         │Audio              │         │
         │                   │         │
┌────────▼────────┐    ┌─────▼───┐ ┌───▼───────────┐
│                 │    │         │ │               │
│  Bidirectional  │    │  Nova2  │ │  Call State   │
│  Audio Stream   │    │  STT    │ │   Manager     │
│                 │    │         │ │               │
└────────┬────────┘    └─────┬───┘ └───┬───────────┘
         │                   │         │
         │                   │         │
┌────────▼────────┐    ┌─────▼───┐ ┌───▼───────────┐
│                 │    │         │ │               │
│     Nova2       │    │   LLM   │ │ UK Regulations│
│      TTS        │    │ Service │ │   Compliance  │
│                 │    │         │ │               │
└─────────────────┘    └─────────┘ └───────────────┘
```

## Key Components Implementation

### 1. STT Integration

The STT integration module (`stt_integration.py`) establishes a WebSocket connection to Nova's STT service, enabling real-time transcription of caller audio. It:

- Converts audio from Twilio's format to the format expected by Nova STT
- Streams audio chunks to Nova's STT service
- Processes transcription results and triggers callbacks
- Manages WebSocket connections for multiple concurrent calls

Key implementation details:

```python
# Establishing WebSocket connection to Nova STT
async def _connect_to_nova_stt(self, call_sid: str) -> websockets.WebSocketClientProtocol:
    """Establish WebSocket connection to Nova STT service"""
    try:
        ws = await websockets.connect(NOVA_STT_WS_URL)
        # Send initialization message with call_sid
        await ws.send(json.dumps({
            "type": "init",
            "call_sid": call_sid,
            "format": {
                "sample_rate": TWILIO_SAMPLE_RATE,
                "channels": TWILIO_CHANNELS,
                "bit_depth": TWILIO_BIT_DEPTH
            }
        }))
        
        # Store the connection
        self.active_connections[call_sid] = ws
        logger.info(f"Connected to Nova STT for call {call_sid}")
        
        # Start listening for transcriptions in the background
        asyncio.create_task(self._listen_for_transcriptions(call_sid, ws))
        
        return ws
    except Exception as e:
        logger.error(f"Failed to connect to Nova STT: {e}")
        raise
```

We use an asynchronous approach to handle multiple concurrent calls efficiently, and background tasks to process incoming transcriptions without blocking the main audio streaming.

### 2. Telemarketer Server

The telemarketer server (`telemarketer_server.py`) is implemented using FastAPI and handles:

- Twilio webhook endpoints for inbound calls
- WebSocket connections for bidirectional audio streaming
- Call state management
- LLM action endpoints
- Outbound call queueing and rate limiting

#### Twilio Integration

We use Twilio's `<Connect><Stream>` TwiML verb to establish bidirectional WebSocket communication:

```python
@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Handle Twilio voice webhook for inbound calls"""
    form_data = await request.form()
    
    # Get call data
    call_sid = form_data.get("CallSid", "")
    from_number = form_data.get("From", "")
    to_number = form_data.get("To", "")
    
    # Create a call state
    await call_state_manager.create_call_state(call_sid, {
        "is_outbound": False,
        "script": "",
        "from_number": from_number,
        "to_number": to_number
    })
    
    # Generate TwiML response
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"{SERVER_BASE_URL}/twilio/stream")
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")
```

#### WebSocket Handling

The WebSocket endpoint handles bidirectional audio streaming with Twilio:

```python
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """Handle Twilio WebSocket streaming connection"""
    await websocket.accept()
    
    call_sid = None
    
    try:
        # First message should contain connection info
        message = await websocket.receive_text()
        data = json.loads(message)
        
        # Extract the call SID
        call_sid = data.get("start", {}).get("callSid")
        
        # Store the WebSocket connection and start processing
        active_ws_connections[call_sid] = websocket
        await stt_processor.start_call_processing(call_sid, handle_transcription)
        
        # Trigger the initial greeting
        state = await call_state_manager.get_call_state(call_sid)
        if state:
            next_action = state.transition(ConversationEvent.CALL_CONNECTED)
            await handle_next_action(call_sid, next_action)
        
        # Loop to process incoming audio
        while True:
            message = await websocket.receive()
            
            # Handle binary media (audio)
            if "bytes" in message:
                audio_data = message["bytes"]
                if audio_data:
                    # Process audio chunk with STT
                    await stt_processor.process_audio_chunk(call_sid, audio_data)
            
            # Handle text messages
            elif "text" in message:
                # Process JSON messages from Twilio
                data = json.loads(message["text"])
                if data.get("event") == "stop":
                    break
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        # Clean up resources when connection ends
        if call_sid:
            if call_sid in active_ws_connections:
                del active_ws_connections[call_sid]
            await stt_processor.end_call_processing(call_sid)
```

#### Call Termination

A critical feature is the ability for the LLM to signal the end of a conversation:

```python
async def terminate_call(call_sid: str, farewell_message: Optional[str] = None):
    """Terminate a call with Twilio and clean up resources"""
    try:
        # Check if we have an active WebSocket
        if call_sid not in active_ws_connections:
            return False
            
        ws = active_ws_connections[call_sid]
        
        # If we have a farewell message, speak it before ending
        if farewell_message:
            streaming_audio = telemarketer.speak(farewell_message, stream=True, play=False)
            await ws.send_json({"type": "audio_start", "text": farewell_message})
            for chunk in streaming_audio:
                await ws.send_bytes(chunk)
            await ws.send_json({"type": "audio_end"})
            await asyncio.sleep(1)
        
        # Send end_call signal to Twilio
        await ws.send_json({"type": "end_call"})
        
        # Clean up resources
        state = await call_state_manager.get_call_state(call_sid)
        if state:
            state.transition(ConversationEvent.END_CALL)
            await call_state_manager.save_call_state(call_sid)
        await stt_processor.end_call_processing(call_sid)
        
        # Try to terminate call via Twilio API as backup
        if twilio_client:
            try:
                twilio_client.calls(call_sid).update(status="completed")
            except Exception as e:
                logger.warning(f"Could not terminate call via Twilio API: {e}")
                
        # Remove from active connections
        if call_sid in active_ws_connections:
            del active_ws_connections[call_sid]
            
        return True
    except Exception as e:
        logger.error(f"Error terminating call {call_sid}: {e}")
        return False
```

### 3. LLM Integration

The system provides an endpoint for the LLM to control the conversation:

```python
@app.post("/api/llm-action")
async def llm_action(action_request: LLMAction):
    """Endpoint for the LLM to signal actions like ending a conversation"""
    call_sid = action_request.call_sid
    action = action_request.action.lower()
    
    # Handle different action types
    if action == "end_conversation":
        # Get farewell message if provided, otherwise use default
        farewell_message = action_request.message or "Thank you for your time. Goodbye!"
        success = await terminate_call(call_sid, farewell_message)
        
        return {
            "success": success,
            "action": action,
            "call_sid": call_sid,
            "message": "Call termination request processed" if success else "Failed to terminate call"
        }
    
    # Other actions can be implemented here
    return {"success": False, "action": action, "message": f"Unknown action: {action}"}
```

This allows the LLM to end conversations or potentially perform other actions in the future.

## API Endpoints and Documentation

### Twilio Webhook Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/twilio/voice` | POST | Handles incoming calls to Twilio numbers |
| `/twilio/status-callback` | POST | Receives call status updates |
| `/twilio/stream` | WebSocket | Bidirectional audio streaming with Twilio |

### Call Management API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/outbound-call` | POST | Queue an outbound call |
| `/api/active-calls` | GET | List active calls |
| `/api/call/{call_sid}` | GET | Get details for a specific call |
| `/api/call-queue` | GET | Get call queue status |

### Compliance API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/uk-regulations/can-call/{phone_number}` | GET | Check if a number can be called |
| `/api/uk-regulations/call-history/{phone_number}` | GET | Get call history for a number |

### LLM Control API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/llm-action` | POST | Endpoint for LLM to signal actions |

### Request and Response Examples

#### Example: Queuing an Outbound Call

Request:
```json
POST /api/outbound-call
{
  "phone_number": "+441234567890",
  "script": "Hello, I'm calling about your recent inquiry about our AI services.",
  "priority": 3,
  "caller_id": "+441234567891"
}
```

Response:
```json
{
  "success": true,
  "call_id": "f8c3de3d-1234-4567-8901-23456789abcd",
  "position": 1,
  "estimated_wait": "2 seconds"
}
```

#### Example: LLM Ending a Conversation

Request:
```json
POST /api/llm-action
{
  "call_sid": "CA1234567890abcdef1234567890abcdef",
  "action": "end_conversation",
  "reason": "User declined offer",
  "message": "I understand you're not interested at this time. Thank you for chatting with me today. Goodbye!"
}
```

Response:
```json
{
  "success": true,
  "action": "end_conversation",
  "call_sid": "CA1234567890abcdef1234567890abcdef",
  "message": "Call termination request processed"
}
```

## Call Flow and State Management

The system uses a state machine to manage conversation flow:

1. **Greeting**: Initial greeting when call connects
2. **Introduction**: Introduction of product/service
3. **Questions**: Series of questions to qualify the prospect
4. **Objection Handling**: Address concerns or objections
5. **Closing**: Request for desired action
6. **Farewell**: End the call politely

Each state transition is triggered by:
- User responses (via STT)
- LLM decisions
- Timer/scheduling events

The state machine is implemented in the `CallStateMachine` class in `call_state_manager.py`.

## UK Regulations Compliance

The system enforces UK telemarketing regulations through the `UKCallRegulator` class:

1. **Time Restrictions**:
   - Weekdays: 8am to 9pm
   - Weekends: 9am to 5pm
   - No calls on UK bank holidays

2. **Frequency Restrictions**:
   - Maximum 3 calls to same number in 24 hours
   - Maximum 6 calls to same number in 7 days

3. **TPS Registry Compliance**:
   - Option to check against Telephone Preference Service

4. **Call Recording and Logging**:
   - Maintains detailed call history
   - Records call outcomes and durations

## WebSocket Communication Protocol

The WebSocket communication between our server and Twilio follows this sequence:

1. **Connection Initialization**:
   ```json
   {
     "start": {
       "callSid": "CA1234567890abcdef",
       "streamSid": "MZ1234567890abcdef",
       "accountSid": "AC1234567890abcdef",
       "callStatus": "in-progress"
     }
   }
   ```

2. **Audio Data**:
   - Binary audio data chunks from Twilio to our server
   - Our server sends audio chunks back to Twilio

3. **Control Messages**:
   ```json
   {"type": "audio_start", "text": "Hello, how can I help you?"}
   {"type": "audio_end"}
   {"type": "end_call"}
   ```

4. **Twilio Events**:
   ```json
   {"event": "mark", "mark": {"name": "waitingforresponse"}}
   {"event": "stop", "streamSid": "MZ1234567890abcdef"}
   ```

## Testing and Development

### Local Development Setup

1. Set up environment variables:
   ```bash
   export TWILIO_ACCOUNT_SID="your_account_sid"
   export TWILIO_AUTH_TOKEN="your_auth_token"
   export TWILIO_DEFAULT_FROM="+1234567890"
   export SERVER_BASE_URL="https://your-server-url.com"
   ```

2. Run the server:
   ```bash
   python -m Nova2.app.telemarketer_server
   ```

3. For local testing, use ngrok to expose your server:
   ```bash
   ngrok http 8000
   ```

4. Configure your Twilio number's webhook to point to your ngrok URL:
   - Voice URL: `https://your-ngrok-domain.ngrok.io/twilio/voice`
   - Status Callback URL: `https://your-ngrok-domain.ngrok.io/twilio/status-callback`

## Future Development Roadmap

### Conversation Management Enhancements

1. **Advanced State Management**:
   - Implement more complex conversation flows
   - Add branching logic based on user responses
   - Support for multi-turn conversations with context

2. **Script Templates**:
   - Create reusable conversation templates
   - Support for dynamic script variables
   - A/B testing for different scripts

### LLM Integration Improvements

1. **Real-time Intent Recognition**:
   - Integrate LLM for intent classification
   - Sentiment analysis to adapt conversation flow
   - Entity extraction for personalization

2. **Learning and Adaptation**:
   - Feedback loops to improve conversation effectiveness
   - Performance metrics tracking
   - Automated script optimization

### Dashboard Integration

1. **Admin Dashboard**:
   - Real-time call monitoring
   - Call analytics and reporting
   - Script management interface
   - Agent intervention capabilities

2. **API Enhancements**:
   - Bulk call scheduling
   - Campaign management
   - Integration with CRM systems

### Technical Improvements

1. **Performance Optimization**:
   - Caching for frequently used TTS responses
   - Load balancing for high call volumes
   - Optimized audio processing

2. **Security Enhancements**:
   - Authentication for API endpoints
   - Encryption for sensitive data
   - Role-based access control

## Common Issues and Troubleshooting

### WebSocket Connection Issues

If you encounter issues with the WebSocket connection:
- Ensure your server is accessible from the internet
- Check that the URL in the TwiML response is correct
- Verify your ngrok tunnel is running
- Check for firewall restrictions

### Audio Quality Issues

For audio quality problems:
- Ensure proper audio format conversion
- Check for packet loss or latency issues
- Verify Nova TTS and STT settings

### Call Termination Issues

If calls are not ending properly:
- Verify the WebSocket connection is still active
- Check for errors in the terminate_call function
- Ensure the "end_call" message is being sent correctly

## Conclusion

This initial implementation provides a solid foundation for the Nova2 Telemarketer System. With bidirectional streaming, real-time STT and TTS integration, and LLM-controlled conversations, we have built the core infrastructure needed for advanced AI-powered voice applications.

Future development should focus on enhancing the conversation management capabilities, improving the LLM integration, and building a comprehensive dashboard for monitoring and management.

---

## Next Steps

1. Refine and test the existing implementation with real-world calls
2. Develop more sophisticated conversation flows
3. Integrate with a production LLM for conversation management
4. Implement analytics and performance tracking
5. Build an admin dashboard for call management
