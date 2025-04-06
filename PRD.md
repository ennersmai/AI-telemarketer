# Product Requirements Document: AI Telemarketer - Version 1.2

**Date:** March 22, 2025

**Author:** Gemini (AI Assistant)

**1. Introduction**

1.1. **Purpose:** This document outlines the product requirements for version 1.2 of our AI Telemarketer, a system designed to automate outbound telemarketing calls using advanced conversational AI. This version focuses on establishing the core architecture and integrating the Nova2 framework with the Twilio telephony platform.

1.2. **Goals:**
    * To create a functional AI agent capable of conducting telemarketing calls.
    * To seamlessly integrate the Nova2 conversational AI framework with the Twilio telephony service.
    * To establish real-time audio communication between the AI agent and call recipients.
    * To lay the foundation for implementing a specific telemarketing script.

1.3. **Target User:** The primary user of this document is the development team responsible for building and deploying the AI Telemarketer.

**2. High-Level Overview**

The AI Telemarketer will leverage the Nova2 framework for its conversational intelligence (including LLM, TTS, STT, and context management) and Twilio for handling the telephony aspects (making calls, managing call states). A dedicated `twilio_manager.py` script will orchestrate the interaction between these two systems.

**3. Functional Requirements**

3.1. **Twilio Integration via `twilio_manager.py`:**
    * The system shall include a separate Python script (`twilio_manager.py`) responsible for interacting directly with the Twilio API.
    * The `twilio_manager` shall be capable of:
        * Managing a list of phone numbers for outbound calls.
        * Initiating outbound phone calls using the Twilio API.
        * Monitoring the state of calls (e.g., ringing, connected, ended) using the Twilio API's event mechanisms (e.g., webhooks).
        * Establishing a WebSocket connection with the Nova2 framework upon successful call connection.
        * Forwarding real-time audio from the caller (received by Twilio) to Nova2 via the WebSocket.
        * Receiving real-time audio output from Nova2 (TTS) via the WebSocket and sending it to Twilio for playback to the caller.
        * Handling call termination signals from both Nova2 and Twilio.

3.2. **Nova2 API and WebSocket Functionality:**
    * The Nova2 framework integration shall include:
        * **API Endpoint for Call Initiation:** A lightweight API endpoint (e.g., using Flask or FastAPI) that the `twilio_manager` can call to signal the start of a new conversation and provide essential call metadata (e.g., Twilio Call SID, caller ID, campaign ID).
        * **WebSocket Endpoint for Audio Streaming:** Functionality to establish a WebSocket endpoint within Nova2 to receive the real-time audio stream from the `twilio_manager`.
        * **Speech-to-Text (STT) Integration:** Seamless integration of Nova2's STT capabilities to transcribe the incoming audio stream from the caller.
        * **Conversational Workflow (LLM and Context):** Leveraging Nova2's LLM and **context management** to process the transcribed text, maintain conversation state, and generate appropriate responses based on the telemarketing script.
        * **Text-to-Speech (TTS) Integration:** Seamless integration of Nova2's TTS capabilities to convert the generated responses into an audio stream.
        * **WebSocket Output for Audio:** Functionality to stream the generated TTS audio back to the `twilio_manager` via the WebSocket connection.
        * **Call End Signaling:** A mechanism within Nova2 to signal to the `twilio_manager` when the conversation should be terminated (e.g., upon reaching the end of the script or based on user input).

3.3. **Telemarketing Script Implementation:**
    * The system shall allow for the implementation of a specific telemarketing script within Nova2.
    * The script implementation shall leverage Nova2's conversational pathways, using prompts, system messages, and context management to guide the LLM's responses.

**4. Non-Functional Requirements**

4.1. **Latency:** The system shall aim for minimal latency in audio transmission and response generation to ensure a natural conversational flow. The target round-trip latency for audio processing and response generation should be within an acceptable range (to be further defined).
4.2. **Reliability:** The integration between Nova2 and Twilio should be reliable and handle potential network issues or API errors gracefully.
4.3. **Scalability (Future Consideration):** While not a primary focus for v1.2, the architecture should be designed with potential future scalability in mind.

**5. High-Level Architecture**

twilio_manager.py (Twilio API Client)
<----> (Bi-directional communication for control and signaling)
Nova2 Framework (Conversational AI)
^ WebSocket (Audio Stream)
|
|
v
  STT Module (within Nova2)
  LLM Module (within Nova2)
  TTS Module (within Nova2)
  Context Management (within Nova2)
  API Endpoint (within Nova2)
^ WebSocket (Audio Stream)
|
|
v
Twilio Platform (Telephony Service)

The `twilio_manager.py` will connect to the Twilio API for call control and establish a WebSocket connection to the Nova2 framework for real-time audio streaming. Nova2 will handle the conversational logic and stream its audio output back to the `twilio_manager` for playback via Twilio.

**6. Twilio Integration Details**

The integration with Twilio will primarily be managed by the `twilio_manager.py`. It will handle the complexities of the Twilio API for initiating and managing calls. The key integration point with Nova2 will be the WebSocket connection for the real-time audio stream, along with a lightweight API endpoint in Nova2 for initial call setup and potentially for signaling the end of the conversation.

**7. Nova2 Integration Details**

Nova2 will need to be extended to include an API endpoint (likely using Flask or FastAPI) to receive call initiation requests from the `twilio_manager`. Furthermore, it will require the implementation of WebSocket handling capabilities to receive and send real-time audio streams. This WebSocket functionality will need to be tightly integrated with Nova2's STT and TTS modules to enable seamless processing of the audio data within the conversational workflow. **Crucially, Nova2 will be responsible for all context management related to the conversation.**

**8. Future Considerations**

* Integration of long-term memory within Nova2 to personalize interactions over multiple calls.
* More sophisticated call flow management and branching logic based on user responses.
* Implementation of error handling and logging mechanisms in both the `twilio_manager` and Nova2.
* Development of a user interface for managing leads and campaigns.