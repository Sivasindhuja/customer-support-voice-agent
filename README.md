# Voice Interview Agent

A realtime AI-powered mock interview agent built using Pipecat, LiveKit, Deepgram, and Gemini.

The agent conducts a voice-based technical interview by:

* asking interview questions,
* listening to candidate responses using speech-to-text,
* evaluating responses against ideal reference answers,
* asking follow-up questions,
* and generating structured feedback at the end.

---

# Features

* Realtime voice interaction
* Speech-to-Text using Deepgram
* Text-to-Speech using Deepgram Aura
* LLM-powered interviewer using Gemini
* Configurable interview dataset (JSON-based)
* Grounded evaluation using ideal answers + keywords
* Follow-up question handling
* Final structured feedback generation
* LiveKit realtime audio transport
* Latency metrics logging

---

# Tech Stack

* Python 3.11
* Pipecat
* LiveKit
* Deepgram
* Gemini 2.5 Flash

---
# Architecture Note

This document contains the trade offs and engineering descisions

https://docs.google.com/document/d/12YcvrZX4_wZvH55hoHcKxt2DuKai6_ndk6qvlkAH7lQ/edit?usp=sharing

# Project Structure

```text
voice-agent/
│
├── app.py
├── config.json
├── interview_qa.json
├── .env
├── requirements.txt
└── interview_feedback.txt
```

---

# Setup

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd voice-agent
```

---

## 2. Create Virtual Environment

```bash
python -m venv env
```

Activate environment:

### Windows

```bash
env\Scripts\activate
```

### Mac/Linux

```bash
source env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pipecat-ai livekit-api deepgram-sdk python-dotenv
```

---

# API Keys Required

You need:

* Deepgram API Key
* Gemini API Key
* LiveKit Cloud Credentials

---

# Environment Variables

Create a `.env` file:

```env
DEEPGRAM_API_KEY=your_deepgram_key

GEMINI_API_KEY=your_gemini_key

LIVEKIT_URL=wss://your-project.livekit.cloud

LIVEKIT_API_KEY=your_livekit_api_key

LIVEKIT_API_SECRET=your_livekit_api_secret
```

---

# LiveKit Setup

Create a project in LiveKit Cloud:

https://cloud.livekit.io

Copy:

* WebSocket URL
* API Key
* API Secret

Paste them into `.env`.

---

# Running the Agent

Start the application:

```bash
python app.py
```

You should see:

```text
Connected to mock-interview-room
LiveKitInputTransport started
LiveKitOutputTransport started
```

---

# Joining the Interview Room

Open LiveKit Meet:

https://meet.livekit.io

Enter:

* your LiveKit server URL
* room name:

  ```text
  mock-interview-room
  ```

Join the room.

Once the participant joins:

* the interview starts automatically,
* the agent asks questions,
* listens to answers,
* and generates final feedback.

---

# Configurable Interview Dataset

Interview questions are stored in:

```text
interview_qa.json
```

Example format:

```json
{
  "id": "q1",
  "question_en": "What is a REST API?",
  "ideal_answer": "A REST API allows systems to communicate over HTTP using GET and POST.",
  "keywords": ["HTTP", "GET", "POST"]
}
```

You can:

* add questions,
* edit answers,
* change domains,
  without modifying application logic.

---

# Feedback Generation

At the end of the interview, the system generates:

* strengths,
* improvement areas,
* overall interview impression.

Feedback is:

* spoken by the agent,
* and saved to:

```text
interview_feedback.txt
```

---

# Current Limitations

* Uses fixed wait times for responses
* No multilingual support yet
* No vector retrieval layer
* Follow-up handling is basic
* Designed as a prototype/demo system

---

# Architecture Overview

```text
Candidate Speech
        ↓
Deepgram STT
        ↓
Pipecat Pipeline
        ↓
Gemini LLM
        ↓
Deepgram TTS
        ↓
Candidate Audio Output
```

The interview is grounded using:

* ideal answers
* keywords
* configurable JSON datasets

---

# Latency Metrics

The system logs:

* ASR latency
* LLM time-to-first-token
* TTS time-to-first-byte

Useful for realtime voice agent optimization.

---

# Demo Flow

1. Candidate joins LiveKit room
2. Agent asks interview question
3. Candidate answers verbally
4. Agent asks follow-up or next question
5. Final feedback is generated

---

# Future Improvements

* Real semantic retrieval layer
* Dynamic turn detection
* Better follow-up reasoning
* Multilingual interviews
* Web frontend
* Interview scoring dashboard

---

# License

MIT License

```
```
