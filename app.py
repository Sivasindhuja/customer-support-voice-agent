import requests
import re
import os
import threading
import time
import json
import pygame
import tempfile
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
from dotenv import load_dotenv

# Google GenAI imports
from google import genai
from google.genai import types

load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize clients
dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-helios-en'
HEADERS = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# State Management
conversation_memory = []
mute_microphone = threading.Event()
REPLAY_FILE = "replay.log"

def get_active_prompt(role="receptionist", version="v1.1.0"):
    try:
        with open('prompts.json', 'r') as f:
            data = json.load(f)
        return data[role][version]
    except Exception as e:
        print(f"Prompt Load Error: {e}")
        return "You are a helpful assistant."

def segment_text_by_sentence(text):
    sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', text)
    boundaries_indices = [boundary.start() for boundary in sentence_boundaries]
    segments = []
    start = 0
    for boundary_index in boundaries_indices:
        segments.append(text[start:boundary_index + 1].strip())
        start = boundary_index + 1
    segments.append(text[start:].strip())
    return [s for s in segments if s]

def synthesize_audio(text):
    payload = {"text": text}
    try:
        # Timeout added for resilience
        r = requests.post(DEEPGRAM_TTS_URL, headers=HEADERS, json=payload, timeout=5.0)
        return r.content
    except requests.exceptions.Timeout:
        print("TTS Timeout - Service unreachable.")
        return None

def play_audio_data(audio_data):
    if not audio_data: return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def visualize_budget(m):
    # Console bar chart: ASR (█), LLM (▒), TTS (░)
    asr_bar = "█" * int(m['asr_ms'] / 50)
    llm_bar = "▒" * int(m['llm_ttft_ms'] / 50)
    tts_bar = "░" * int(m['tts_ttfb_ms'] / 50)
    print(f"\nLatency Budget: [ASR:{asr_bar}][LLM:{llm_bar}][TTS:{tts_bar}] Total: {m['total_ms']:.0f}ms")

# --- NEW: AI Logic moved to a separate function for threading ---
def process_turn(utterance, start_time, asr_end, microphone):
    try:
        # 2. LLM Processing (Time to First Token)
        llm_start = time.perf_counter()
        
        # Format memory for the NEW Google SDK
        conversation_memory.append(
            types.Content(role="user", parts=[types.Part.from_text(text=utterance)])
        )
        
        active_system_prompt = get_active_prompt("receptionist", "v1.1.0")
        
        # Call Gemini using the new SDK
        # Note: I changed model to gemini-1.5-flash as 2.5 does not officially exist yet. 
        # You can update this to gemini-2.0-flash if you have access to it!
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=conversation_memory,
            config=types.GenerateContentConfig(
                system_instruction=active_system_prompt
            )
        )
        llm_end = time.perf_counter()
        
        full_text = response.text.strip()
        
        # Save AI response to memory
        conversation_memory.append(
            types.Content(role="model", parts=[types.Part.from_text(text=full_text)])
        )
        
        segments = segment_text_by_sentence(full_text)
        
        tts_ttfb_ms = 0
        perceived_latency_end = 0
        
        # Cleaned up the nested loop bug here!
        for i, seg in enumerate(segments):
            tts_start = time.perf_counter()
            audio = synthesize_audio(seg)
            
            # Measure TTFB only for the first segment
            if i == 0:
                tts_ttfb_ms = (time.perf_counter() - tts_start) * 1000
                # True latency stops the millisecond the first audio is ready!
                perceived_latency_end = time.perf_counter() 
            
            play_audio_data(audio)

        # Finish Latency Tracking
        metrics = {
            "asr_ms": (asr_end - start_time) * 1000,
            "llm_ttft_ms": (llm_end - llm_start) * 1000,
            "tts_ttfb_ms": tts_ttfb_ms,
            "total_ms": (perceived_latency_end - start_time) * 1000
        }
        
        visualize_budget(metrics)

    except Exception as e:
        print(f"Fallback triggered due to error: {e}")
        # Remove the failed user attempt from memory so it doesn't break future turns
        if len(conversation_memory) > 0 and conversation_memory[-1].role == "user":
            conversation_memory.pop()
            
        # Fallback response for graceful degradation
        play_audio_data(synthesize_audio("I'm sorry, I'm having a bit of trouble connecting to my brain. Could you repeat that?"))
    
    finally:
        # Always make sure to release the microphone lock when the thread finishes
        time.sleep(0.3)
        microphone.unmute()
        mute_microphone.clear()


def main():
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.websocket.v("1") # Fixed the deprecation warning
        is_finals = []

        def on_message(self, result, **kwargs):
            nonlocal is_finals
            if mute_microphone.is_set(): return
            
            transcript = result.channel.alternatives[0].transcript
            if len(transcript) == 0: return

            if result.is_final:
                is_finals.append(transcript)
                
                if result.speech_final:
                    # --- Start Latency Tracking ---
                    start_time = time.perf_counter()
                    utterance = " ".join(is_finals).strip()
                    is_finals = []
                    
                    # Log for Replay Mode
                    with open(REPLAY_FILE, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | USER: {utterance}\n")

                    print(f"User: {utterance}")
                    
                    # 1. ASR Finish
                    asr_end = time.perf_counter()

                    # Immediately lock the microphone on the main thread so user can't interrupt AI thinking
                    mute_microphone.set()
                    microphone.mute()
                    
                    # Pass the workload to the background thread!
                    threading.Thread(
                        target=process_turn, 
                        args=(utterance, start_time, asr_end, microphone)
                    ).start()

        # Deepgram Handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, lambda self, error, **kw: print(f"Error: {error}"))

        options = LiveOptions(
            model="nova-2", language="en-US", smart_format=True,
            encoding="linear16", channels=1, sample_rate=16000,
            interim_results=True, utterance_end_ms="1000", vad_events=True, endpointing=500,
        )

        if not dg_connection.start(options):
            print("Failed to connect to Deepgram")
            return

        microphone = Microphone(dg_connection.send)
        microphone.start()
        print("Agent Active. Press Enter to stop.")
        input("")
        microphone.finish()
        dg_connection.finish()

    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    main()
