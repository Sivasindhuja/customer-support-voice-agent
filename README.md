#  Voice AI Mock Interviewer

A real-time conversational AI system that conducts technical mock interviews through voice, evaluates candidate responses against predefined technical expectations, asks contextual follow-up questions, and generates structured interview feedback.

> **Status:** Engineering Prototype

---

## Why This Project?

Technical interviews evaluate much more than technical knowledge. Candidates are expected to communicate their ideas clearly, explain trade-offs, and think aloud under pressure. While coding platforms help practice problem solving, they rarely simulate the conversational nature of a real interview.

This project explores how a Voice AI system can provide a realistic interview experience by combining speech recognition, large language models, and speech synthesis into a low-latency conversational pipeline.

---

## Features

-  Real-time voice conversations
-  Grounded technical interviewer using Gemini
-  Context-aware follow-up questions
-  Automated interview feedback
-  Configurable interview dataset
-  Latency metrics (STT, LLM TTFT, TTS TTFB)
-  Configurable prompts and interview behavior

---

## System Overview

```
Candidate
     │
     ▼
 LiveKit
     │
     ▼
 Deepgram STT
     │
     ▼
 Gemini
     │
     ▼
 Deepgram TTS
     │
     ▼
 Candidate
```

---

## How It Works

1. The candidate joins the interview room.
2. The interviewer asks a technical question.
3. The candidate answers using voice.
4. Speech is transcribed using Deepgram STT.
5. Gemini evaluates the response using the current question, ideal answer, and expected concepts.
6. The interviewer asks a follow-up question or proceeds to the next question.
7. After the interview, structured feedback is generated and saved.

---

## Repository Structure

```
Voice-AI-Mock-Interviewer/

├── app.py
├── config.json
├── interview_qa.json
├── requirements.txt
├── README.md
└── docs/
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<username>/Voice-AI-Mock-Interviewer.git

cd Voice-AI-Mock-Interviewer
```

### 2. Create a virtual environment

```bash
python -m venv env
```

Activate the environment.

**Windows**

```bash
env\Scripts\activate
```

**macOS/Linux**

```bash
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file.

```env
DEEPGRAM_API_KEY=

GEMINI_API_KEY=

LIVEKIT_URL=

LIVEKIT_API_KEY=

LIVEKIT_API_SECRET=
```

### 5. Run the application

```bash
python app.py
```

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| Voice Pipeline | Pipecat |
| Transport | LiveKit |
| Speech Recognition | Deepgram STT |
| Speech Synthesis | Deepgram Aura |
| LLM | Gemini 2.0 Flash |
| Configuration | JSON |

---

## Current Limitations

- Fixed interview dataset
- Basic follow-up strategy
- Fixed response timeout
- No persistent candidate memory
- Prototype implementation

---

## Documentation

The README provides a high-level overview of the project.

Detailed engineering documentation covering architecture, design decisions, prompt engineering, latency analysis, and future roadmap will be available on **GitHub Pages**.

> 🚧 Documentation website coming soon.

---

## Future Work

- Streaming LLM responses
- Streaming TTS playback
- Adaptive interviewer memory
- Retrieval-backed question management
- Recruiter dashboard
- Interview analytics
- Docker support

---

## Acknowledgements

This project was built to explore the engineering challenges involved in developing real-time Voice AI systems, including conversational state management, grounded LLM evaluation, prompt engineering, and latency optimization.